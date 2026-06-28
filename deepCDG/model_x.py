import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) wrapper for nn.Linear.
    Allows cancer-specific adaptation of key GCN projection weights.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super(LoRALinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank
        self.enabled = False
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_lora(self, enabled=True):
        self.enabled = enabled

    def forward(self, x):
        out = self.linear(x)
        if self.enabled:
            lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale
            out = out + lora_out
        return out


class StageConditionedPPI(nn.Module):
    """
    Stage-Conditioned Dynamic PPI Gating.
    Computes stage-dependent edge weights to model context-specific PPI dynamics.
    """
    def __init__(self, node_dim, stage_dim=16):
        super(StageConditionedPPI, self).__init__()
        self.stage_emb = nn.Linear(1, stage_dim)
        # Gating network taking concatenated node embeddings and stage embedding
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2 + stage_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, stage):
        # x: [num_nodes, node_dim]
        # edge_index: [2, num_edges]
        # stage: [1] (cohort stage metadata value, e.g., 1.0, 2.0, 3.0, 4.0)
        
        # Embed the stage metadata
        s_emb = self.stage_emb(stage.view(1, 1))  # [1, stage_dim]
        s_emb_expanded = s_emb.expand(edge_index.shape[1], -1)  # [num_edges, stage_dim]
        
        # Get source and destination node embeddings
        u = x[edge_index[0]]  # [num_edges, node_dim]
        v = x[edge_index[1]]  # [num_edges, node_dim]
        
        # Concatenate and pass to gating network
        gate_input = torch.cat([u, v, s_emb_expanded], dim=1)  # [num_edges, node_dim * 2 + stage_dim]
        edge_weight = self.gate(gate_input).squeeze(-1)  # [num_edges]
        
        return edge_weight


class CausalGNNGating(nn.Module):
    """
    Causal Feature Selector.
    Learns a feature-wise mask to select causal features and filter out spurious correlations.
    """
    def __init__(self, dim_hidden):
        super(CausalGNNGating, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [num_nodes, dim_hidden]
        mask = self.gate(x)
        x_causal = x * mask
        return x_causal, mask


class deepCDGX(nn.Module):
    def __init__(self, args, num_omics=6, omics_in_dim=16, dim_hidden=48, dim_hidden2=200, lora_rank=4):
        super(deepCDGX, self).__init__()
        self.args = args
        self.num_omics = num_omics
        self.omics_in_dim = omics_in_dim
        self.dim_hidden = dim_hidden
        self.dim_hidden2 = dim_hidden2
        self.dropout = args.dropout
        self.act = torch.relu

        # 1. 6-Omics Input Encoders (Mutation, Expression, Methylation + Simulated CNV, miRNA, Proteomics)
        # Using GCNConv inside Encoders
        self.encoders = nn.ModuleList([
            EncoderX(omics_in_dim, dim_hidden, self.dropout, self.act)
            for _ in range(num_omics)
        ])

        # Deterministic projections to generate CNV, miRNA, Proteomics from original 3 omics if input has 3 omics
        # original features: Mutation (0:16), Expression (16:32), Methylation (32:48)
        # CNV is generated from Mutation + Methylation
        # miRNA is generated from Expression + Methylation
        # Proteomics is generated from Expression
        self.cnv_generator = nn.Linear(32, omics_in_dim)
        self.mirna_generator = nn.Linear(32, omics_in_dim)
        self.proteomics_generator = nn.Linear(16, omics_in_dim)

        # 2. Multi-Head Cross-Attention (MHCA) for Omics Fusion
        # Treats the 6 omics types as sequence tokens
        self.mhca = nn.MultiheadAttention(embed_dim=dim_hidden, num_heads=4, dropout=self.dropout)
        self.fc_fusion = nn.Linear(num_omics * dim_hidden, dim_hidden)

        # 3. Stage-Conditioned Dynamic PPI Gating
        self.dynamic_ppi = StageConditionedPPI(node_dim=dim_hidden)

        # 4. Causal Feature Selection Gate
        self.causal_gating = CausalGNNGating(dim_hidden)

        # 5. Shared Project and Classifier with LoRA support
        self.project = GCNConvX(dim_hidden, 100, lora_rank)
        self.classifier = ClassifierX(100, dim_hidden2, lora_rank, self.dropout, self.act)

    def set_lora(self, enabled=True):
        """Enable or disable cancer-specific LoRA adapters."""
        self.project.set_lora(enabled)
        self.classifier.set_lora(enabled)

    def generate_6_omics(self, x):
        """
        Generates 6-omics features.
        Supports:
          - 64-column inputs (MF, METH, GE, CNA)
          - 48-column inputs (MF, METH, GE) - CNA generated dynamically
          - <= 4 column inputs (type-specific)
        """
        if x.shape[1] <= 4:
            # Type-specific evaluation
            mut = x[:, 0:1].repeat(1, 16)
            meth = x[:, 1:2].repeat(1, 16)
            exp = x[:, 2:3].repeat(1, 16)
            if x.shape[1] == 4:
                cna = x[:, 3:4].repeat(1, 16)
            else:
                cna = self.cnv_generator(torch.cat([mut, meth], dim=1))
        elif x.shape[1] < 64:
            # 48-column input (traditional deepCDG feature shape)
            mut = x[:, 0:16]
            meth = x[:, 16:32]
            exp = x[:, 32:48]
            cna = self.cnv_generator(torch.cat([mut, meth], dim=1))
        else:
            # Full 64-column pan-cancer dataset (MF, METH, GE, CNA)
            mut = x[:, 0:16]
            meth = x[:, 16:32]
            exp = x[:, 32:48]
            cna = x[:, 48:64]

        # Generate miRNA and Proteomics features
        mirna = self.mirna_generator(torch.cat([exp, meth], dim=1))
        prot = self.proteomics_generator(exp)

        return [mut, meth, exp, cna, mirna, prot]

    def forward(self, x, edge_index, stage=None):
        # 1. Drop out edge index during training
        edge_index, _ = dropout_adj(edge_index, p=self.dropout,
                                    force_undirected=True,
                                    num_nodes=x.shape[0],
                                    training=self.training)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Extract or generate 6 omics features
        omics_features = self.generate_6_omics(x)  # list of 6 tensors, each [num_nodes, 16]

        # 3. GCN Encoder for each omics
        # Encoders extract local neighbor information
        omics_embeddings = []
        for i, feature in enumerate(omics_features):
            emb = self.encoders[i](feature, edge_index)  # [num_nodes, dim_hidden]
            omics_embeddings.append(emb)

        # 4. Multi-Head Cross-Attention Fusion
        # Stack as sequence: [num_omics, num_nodes, dim_hidden]
        stacked_embs = torch.stack(omics_embeddings, dim=0)
        # MHCA (Query, Key, Value are the same for self-attention across modalities)
        attn_out, _ = self.mhca(stacked_embs, stacked_embs, stacked_embs)  # [num_omics, num_nodes, dim_hidden]
        
        # Flatten and project to merge omics
        attn_out = attn_out.transpose(0, 1)  # [num_nodes, num_omics, dim_hidden]
        fused_features = attn_out.reshape(x.shape[0], -1)  # [num_nodes, num_omics * dim_hidden]
        fused_emb = self.fc_fusion(fused_features)  # [num_nodes, dim_hidden]

        # 5. Stage-Conditioned Dynamic PPI Weights
        if stage is None:
            stage = torch.tensor(1.0, device=x.device)  # Default stage: early/medium
        edge_weight = self.dynamic_ppi(fused_emb, edge_index, stage)

        # 6. Causal Feature Gating
        causal_emb, causal_mask = self.causal_gating(fused_emb)

        # 7. Project and Classify using Dynamic Adjacency Weights
        projected = self.project(causal_emb, edge_index, edge_weight)
        projected = F.dropout(projected, p=self.dropout, training=self.training)
        pred = self.classifier(projected, edge_index, edge_weight)

        # If training, return causal mask L1 loss as well
        if self.training:
            self.causal_loss = torch.mean(torch.abs(causal_mask)) # L1 sparsity penalty
        else:
            self.causal_loss = 0.0

        return pred

    @torch.no_grad()
    def forward_mc(self, x, edge_index, stage=None, num_passes=30):
        """
        Inference using Monte Carlo Dropout to quantify uncertainty.
        Returns:
            mean_prob: Calibrated prediction probabilities [num_nodes, 1]
            variance: Prediction epistemic uncertainty [num_nodes, 1]
        """
        self.eval()
        # Custom loop to enable dropout during evaluation
        def enable_dropout(m):
            if type(m) == nn.Dropout or type(m) == nn.MultiheadAttention:
                m.train()

        self.apply(enable_dropout)

        predictions = []
        for _ in range(num_passes):
            pred = torch.sigmoid(self.forward(x, edge_index, stage))
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [num_passes, num_nodes, 1]
        
        mean_prob = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)

        # Restore eval mode for everything
        self.eval()
        return mean_prob, variance


class EncoderX(nn.Module):
    def __init__(self, in_feat, out_feat, dropout, act):
        super(EncoderX, self).__init__()
        self.conv1 = GCNConv(in_feat, out_feat, add_self_loops=False)
        self.fc = nn.Linear(in_feat, out_feat)
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index):
        x0 = self.act(self.fc(x))
        x = self.conv1(x, edge_index)
        return x0 + x


class GCNConvX(nn.Module):
    """
    GCNConv with LoRA linear weights.
    Supports edge weights for dynamic graph convolutions.
    """
    def __init__(self, in_channels, out_channels, lora_rank=4):
        super(GCNConvX, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=False)
        # Wrap linear projection of GCNConv
        self.lora_proj = LoRALinear(in_channels, out_channels, rank=lora_rank)
        # Override the original weight
        self.conv.lin = self.lora_proj

    def set_lora(self, enabled=True):
        self.lora_proj.set_lora(enabled)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight=edge_weight)


class ClassifierX(nn.Module):
    """
    Residual GCN Classifier with LoRA layers and edge weight support.
    """
    def __init__(self, in_feat, in_hidden, lora_rank, dropout, act):
        super(ClassifierX, self).__init__()
        self.conv1 = GCNConv(in_feat, in_hidden, add_self_loops=False)
        self.conv2 = GCNConv(in_hidden, 1, add_self_loops=False)
        
        self.fc1 = LoRALinear(in_feat, in_hidden, rank=lora_rank)
        self.fc2 = LoRALinear(in_hidden, 1, rank=lora_rank)
        
        self.conv1.lin = self.fc1
        self.conv2.lin = self.fc2
        
        self.act = act
        self.dropout = dropout

    def set_lora(self, enabled=True):
        self.fc1.set_lora(enabled)
        self.fc2.set_lora(enabled)

    def forward(self, x, edge_index, edge_weight=None):
        x0 = self.act(self.fc1(x))
        x_conv = self.act(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x0 + x_conv, p=self.dropout, training=self.training)
        
        x0_out = self.fc2(x)
        x_conv_out = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x0_out + x_conv_out
