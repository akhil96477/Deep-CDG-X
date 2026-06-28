import warnings
import copy
import numpy as np
import time
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from utils import get_ppi, k_folds, set_seed
from model_x import deepCDGX
import argparse
from tqdm import tqdm
from sklearn import metrics

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CPDB',
                    choices=['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015'],
                    help="The dataset to be used.")
parser.add_argument('--device', type=str, default='cuda:0',
                    choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
parser.add_argument('--in_channels', type=int, default=16)
parser.add_argument('--hidden_channel_1', type=int, default=48)
parser.add_argument('--hidden_channel_2', type=int, default=200)
parser.add_argument('--epochs', type=int, default=300, help="Fewer epochs needed due to GCL and Focal Loss convergence")
parser.add_argument('--pretrain_epochs', type=int, default=50, help="Epochs for self-supervised GCL pretraining")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_5_CV_pkl', type=bool, default=False)
parser.add_argument('--times', type=int, default=1, help='Times of 5_CV')
parser.add_argument('--lora_rank', type=int, default=4)
parser.add_argument('--use_gcl', type=bool, default=True, help="Enable Graph Contrastive Learning")
parser.add_argument('--mc_passes', type=int, default=15, help="Number of Monte Carlo dropout forward passes")
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Load data
data = get_ppi(args.dataset, PATH='./PPI_data/')
# Preserve all 64 features (MF, METH, GE, CNA)
data.x = data.x[:, :64]
data = data.to(device)

if args.use_5_CV_pkl:
    with open(f'../data/{args.dataset}_data.pkl', 'rb') as file:
        k_sets = pickle.load(file)
else:
    k_sets = k_folds(data)

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        
        # Apply class balancing alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * loss).mean()

# Graph Contrastive Learning Augmentation & Loss
def augment_graph(x, edge_index, drop_feature_rate=0.1, drop_edge_rate=0.1):
    # Feature masking
    x_aug = x.clone()
    num_features = x.size(1)
    feat_mask = torch.rand(num_features, device=x.device) > drop_feature_rate
    x_aug = x_aug * feat_mask.float().view(1, -1)
    
    # Edge dropping
    num_edges = edge_index.size(1)
    edge_mask = torch.rand(num_edges, device=edge_index.device) > drop_edge_rate
    edge_index_aug = edge_index[:, edge_mask]
    
    return x_aug, edge_index_aug

def info_nce_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Similarity matrix between two augmented views
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    # Target is the diagonal indices (positive pairs)
    targets = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(sim_matrix, targets)
    return loss

@torch.no_grad()
def test_x(model, data, mask, mc_passes=15):
    model.eval()
    # Use Monte Carlo Dropout to obtain calibrated probabilities and uncertainty
    pred_mean, pred_var = model.forward_mc(data.x, data.edge_index, num_passes=mc_passes)
    
    pred_mean_flat = pred_mean[mask].cpu().numpy().squeeze()
    y_true = data.y[mask].cpu().numpy().squeeze()
    
    precision, recall, _ = metrics.precision_recall_curve(y_true, pred_mean_flat)
    fpr, tpr, _ = metrics.roc_curve(y_true, pred_mean_flat)
    
    auc_score = metrics.auc(fpr, tpr)
    auprc_score = metrics.auc(recall, precision)
    
    # Epistemic uncertainty metrics
    mean_uncertainty = pred_var[mask].mean().item()
    
    return auc_score, auprc_score, mean_uncertainty

# Benchmarking Lists
auc_matrix = np.zeros(shape=(args.times, 5))
aupr_matrix = np.zeros(shape=(args.times, 5))
uncertainty_matrix = np.zeros(shape=(args.times, 5))

focal_loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

for run in range(args.times):
    for cv_run in range(5):
        print(f"\n--- Run {run + 1}, CV Fold {cv_run + 1} ---")
        tr_mask, te_mask = k_sets[run][cv_run]
        
        # Instantiate deepCDG-X
        model = deepCDGX(args).to(device)
        
        # Phase 1: Self-Supervised Pretraining using Graph Contrastive Learning (GCL)
        if args.use_gcl and args.pretrain_epochs > 0:
            print("Pretraining with Graph Contrastive Learning...")
            optimizer_pre = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            for epoch in range(args.pretrain_epochs):
                model.train()
                optimizer_pre.zero_grad()
                
                # Create two augmented views
                x_aug1, edge_aug1 = augment_graph(data.x, data.edge_index)
                x_aug2, edge_aug2 = augment_graph(data.x, data.edge_index)
                
                # Get embeddings (from model's intermediate state)
                # For pretraining, we extract the fused embedding before the final projection/classifier
                # Let's perform forward passes on the encoder/fusion portion
                
                # View 1
                omics_features1 = model.generate_6_omics(x_aug1)
                embs1 = [model.encoders[i](f, edge_aug1) for i, f in enumerate(omics_features1)]
                stacked1 = torch.stack(embs1, dim=0)
                attn_out1, _ = model.mhca(stacked1, stacked1, stacked1)
                fused1 = model.fc_fusion(attn_out1.transpose(0, 1).reshape(data.x.shape[0], -1))
                
                # View 2
                omics_features2 = model.generate_6_omics(x_aug2)
                embs2 = [model.encoders[i](f, edge_aug2) for i, f in enumerate(omics_features2)]
                stacked2 = torch.stack(embs2, dim=0)
                attn_out2, _ = model.mhca(stacked2, stacked2, stacked2)
                fused2 = model.fc_fusion(attn_out2.transpose(0, 1).reshape(data.x.shape[0], -1))
                
                # Compute GCL Loss
                gcl_loss = info_nce_loss(fused1, fused2)
                gcl_loss.backward()
                optimizer_pre.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs} - GCL Loss: {gcl_loss.item():.4f}")

        # Phase 2: Supervised Fine-tuning with Focal Loss & Stage Gating
        # Unfreeze all parameters including cancer-specific LoRA adapters
        model.set_lora(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        print("Supervised training with Focal Loss and Stage conditioning...")
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            
            # Simulate stage metadata (dynamic environment)
            # Stage ranges from 1.0 (early) to 4.0 (late stage)
            stage_val = torch.tensor(np.random.uniform(1.0, 4.0), dtype=torch.float32, device=device)
            
            # Forward pass
            pred = model(data.x, data.edge_index, stage=stage_val)
            
            # Supervised focal loss on training fold
            cls_loss = focal_loss_fn(pred[tr_mask], data.y[tr_mask].float())
            
            # Total Loss includes the model's internal causal sparsity loss
            loss = cls_loss + 0.05 * model.causal_loss
            
            loss.backward()
            optimizer.step()
            
        # Phase 3: Evaluation using Monte Carlo Dropout
        auc_s, aupr_s, var_s = test_x(model, data, te_mask, mc_passes=args.mc_passes)
        auc_matrix[run][cv_run] = auc_s
        aupr_matrix[run][cv_run] = aupr_s
        uncertainty_matrix[run][cv_run] = var_s
        
        print(f"Fold Results -> ROC-AUC: {auc_s:.4f}, AUPRC: {aupr_s:.4f}, Mean Uncertainty (MC Var): {var_s:.5f}")

print("\n================== FINAL DEEPCDG-X PERFORMANCE ==================")
print(f"Mean ROC-AUC: {np.mean(auc_matrix):.4f} +/- {np.std(auc_matrix):.4f}")
print(f"Mean AUPRC  : {np.mean(aupr_matrix):.4f} +/- {np.std(aupr_matrix):.4f}")
print(f"Mean Epistemic Uncertainty: {np.mean(uncertainty_matrix):.5f}")
print("=================================================================")
