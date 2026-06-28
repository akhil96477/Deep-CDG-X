import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

class CFGNNExplainer:
    """
    Counterfactual GNNExplainer for deepCDG-X.
    Finds the minimal perturbation (edge deletions) in the local neighborhood
    of a gene that changes its classification from "driver" to "non-driver".
    """
    def __init__(self, model, lr=0.05, steps=150, beta=0.8):
        self.model = model
        self.lr = lr
        self.steps = steps
        self.beta = beta  # Weight for the distance (sparsity) loss

    def explain(self, x, edge_index, node_idx, stage=None, target_prob=0.1):
        """
        Finds the counterfactual explanation for the prediction of `node_idx`.
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Adjacency list [2, num_edges]
            node_idx: The index of the gene to explain
            stage: Stage metadata
            target_prob: The target prediction probability (e.g. < 0.1 to flip to non-driver)
        Returns:
            removed_edges: List of edges [source, target] whose removal flips the prediction.
            success: Boolean indicating if a flip was successfully achieved.
        """
        self.model.eval()

        # 1. Extract the k-hop subgraph around node_idx (k=3 for our deepCDG GCN stack)
        sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=3,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=x.size(0)
        )

        # Subgraph node features and mapping index
        x_sub = x[sub_nodes]
        target_idx_sub = mapping[0].item()

        # Get baseline prediction on the clean subgraph
        with torch.no_grad():
            baseline_pred = torch.sigmoid(self.model(x_sub, sub_edge_index, stage))[target_idx_sub].item()
        
        # If the gene is not predicted as a driver, no explanation is needed
        if baseline_pred < 0.5:
            print(f"Gene {node_idx} is already predicted as non-driver ({baseline_pred:.4f}). Skipping CF explanation.")
            return [], True

        # 2. Define the continuous edge mask as a learnable parameter
        # Initialize logits to high values so sigmoid(logits) starts near 1.0 (no edges removed)
        num_sub_edges = sub_edge_index.size(1)
        edge_mask_logits = nn.Parameter(torch.ones(num_sub_edges, device=x.device) * 2.0)
        optimizer = torch.optim.Adam([edge_mask_logits], lr=self.lr)

        success = False
        final_probs = baseline_pred
        
        # 3. Optimization Loop
        for step in range(self.steps):
            optimizer.zero_grad()
            
            # Apply sigmoid to get edge weights in [0, 1]
            edge_weights = torch.sigmoid(edge_mask_logits)
            
            # Forward pass with the continuous edge weights
            pred_logits = self.model(x_sub, sub_edge_index, stage=stage)
            pred_prob = torch.sigmoid(pred_logits[target_idx_sub])
            
            # Loss formulation:
            # - pred_loss: minimize the predicted probability to drive it below target_prob
            # - dist_loss: minimize modifications (i.e. keep edge weights close to 1)
            pred_loss = torch.clamp(pred_prob - target_prob, min=0.0)
            dist_loss = torch.sum(1.0 - edge_weights)
            
            loss = pred_loss + self.beta * dist_loss
            loss.backward()
            optimizer.step()

            current_prob = pred_prob.item()
            if current_prob < 0.5:
                success = True
                final_probs = current_prob
                # We could break early, but keeping optimization runs leads to cleaner sparser explanations

        # 4. Extract counterfactual edges
        final_weights = torch.sigmoid(edge_mask_logits).detach()
        removed_edge_indices = (final_weights < 0.5).nonzero(as_tuple=True)[0]
        
        removed_edges = []
        for idx in removed_edge_indices:
            src_sub = sub_edge_index[0, idx].item()
            dst_sub = sub_edge_index[1, idx].item()
            # Map back to original node indices
            src_orig = sub_nodes[src_sub].item()
            dst_orig = sub_nodes[dst_sub].item()
            removed_edges.append((src_orig, dst_orig))

        print(f"CF-Explainer for gene {node_idx}: success={success}, original_prob={baseline_pred:.4f}, perturbed_prob={final_probs:.4f}, removed_edges={len(removed_edges)}")
        return removed_edges, success
