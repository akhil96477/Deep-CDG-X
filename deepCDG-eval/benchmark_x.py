import warnings
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import argparse

# Add deepCDG path to import models and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../deepCDG')))

from utils import get_ppi, k_folds, set_seed
# Import the original Net
from model import Net as OriginalNet
# Import the upgraded deepCDG-X
from model_x import deepCDGX

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
parser.add_argument('--epochs', type=int, default=200, help="Epochs for training comparison")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mc_passes', type=int, default=15)
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Load dataset (looking in ../deepCDG/PPI_data/)
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../deepCDG/PPI_data/'))
data = get_ppi(args.dataset, PATH=data_path)
data.x = data.x[:, :64]
data = data.to(device)

k_sets = k_folds(data)

# Focal Loss for deepCDG-X comparison
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
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * loss).mean()

# GCL Augmentation for deepCDG-X comparison
def augment_graph(x, edge_index, drop_feature_rate=0.1, drop_edge_rate=0.1):
    x_aug = x.clone()
    num_features = x.size(1)
    feat_mask = torch.rand(num_features, device=x.device) > drop_feature_rate
    x_aug = x_aug * feat_mask.float().view(1, -1)
    
    num_edges = edge_index.size(1)
    edge_mask = torch.rand(num_edges, device=edge_index.device) > drop_edge_rate
    edge_index_aug = edge_index[:, edge_mask]
    
    return x_aug, edge_index_aug

def info_nce_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim_matrix, targets)

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n=======================================================")
print(f"  deepCDG vs deepCDG-X Benchmarking (Dataset: {args.dataset})")
print(f"=======================================================\n")

# Benchmark original deepCDG
print("Running original deepCDG benchmark...")
orig_auc_list = []
orig_aupr_list = []
orig_time_list = []

for cv_run in range(5):
    tr_mask, te_mask = k_sets[0][cv_run]
    model = OriginalNet(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)
        cls_loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].float())
        cls_loss.backward()
        optimizer.step()
    
    train_time = time.time() - start_time
    orig_time_list.append(train_time)
    
    # Test
    model.eval()
    with torch.no_grad():
        x_out = model(data.x, data.edge_index)
        pred_test = torch.sigmoid(x_out[te_mask]).cpu().numpy()
        y_test = data.y[te_mask].cpu().numpy()
        
    precision, recall, _ = metrics.precision_recall_curve(y_test, pred_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, pred_test)
    auc_val = metrics.auc(fpr, tpr)
    aupr_val = metrics.auc(recall, precision)
    
    orig_auc_list.append(auc_val)
    orig_aupr_list.append(aupr_val)
    print(f"  Fold {cv_run+1} -> AUC: {auc_val:.4f}, AUPRC: {aupr_val:.4f}, Time: {train_time:.2f}s")

orig_params = count_parameters(OriginalNet(args))

# Benchmark deepCDG-X
print("\nRunning deepCDG-X benchmark...")
x_auc_list = []
x_aupr_list = []
x_time_list = []
x_unc_list = []

focal_loss_fn = FocalLoss()

for cv_run in range(5):
    tr_mask, te_mask = k_sets[0][cv_run]
    model = deepCDGX(args).to(device)
    
    start_time = time.time()
    
    # Optional GCL pretraining (10 epochs for quick bench)
    pretrain_epochs = 10
    optimizer_pre = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(pretrain_epochs):
        model.train()
        optimizer_pre.zero_grad()
        x_aug1, edge_aug1 = augment_graph(data.x, data.edge_index)
        x_aug2, edge_aug2 = augment_graph(data.x, data.edge_index)
        
        # View 1
        features1 = model.generate_6_omics(x_aug1)
        embs1 = [model.encoders[i](f, edge_aug1) for i, f in enumerate(features1)]
        fused1 = model.fc_fusion(torch.stack(embs1, dim=0).transpose(0, 1).reshape(data.x.shape[0], -1))
        
        # View 2
        features2 = model.generate_6_omics(x_aug2)
        embs2 = [model.encoders[i](f, edge_aug2) for i, f in enumerate(features2)]
        fused2 = model.fc_fusion(torch.stack(embs2, dim=0).transpose(0, 1).reshape(data.x.shape[0], -1))
        
        gcl_loss = info_nce_loss(fused1, fused2)
        gcl_loss.backward()
        optimizer_pre.step()
        
    # Supervised fine-tuning with Focal Loss & Stage metadata
    model.set_lora(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        stage_val = torch.tensor(1.0, device=device) # baseline stage
        pred = model(data.x, data.edge_index, stage=stage_val)
        cls_loss = focal_loss_fn(pred[tr_mask], data.y[tr_mask].float())
        loss = cls_loss + 0.05 * model.causal_loss
        loss.backward()
        optimizer.step()
        
    train_time = time.time() - start_time
    x_time_list.append(train_time)
    
    # Test using Monte Carlo Dropout
    model.eval()
    pred_mean, pred_var = model.forward_mc(data.x, data.edge_index, num_passes=args.mc_passes)
    pred_mean_flat = pred_mean[te_mask].cpu().numpy().squeeze()
    y_test = data.y[te_mask].cpu().numpy().squeeze()
    
    precision, recall, _ = metrics.precision_recall_curve(y_test, pred_mean_flat)
    fpr, tpr, _ = metrics.roc_curve(y_test, pred_mean_flat)
    
    auc_val = metrics.auc(fpr, tpr)
    aupr_val = metrics.auc(recall, precision)
    unc_val = pred_var[te_mask].mean().item()
    
    x_auc_list.append(auc_val)
    x_aupr_list.append(aupr_val)
    x_unc_list.append(unc_val)
    print(f"  Fold {cv_run+1} -> AUC: {auc_val:.4f}, AUPRC: {aupr_val:.4f}, Uncertainty: {unc_val:.5f}, Time: {train_time:.2f}s")

x_params = count_parameters(deepCDGX(args))

# Final Comparison Output
print("\n=======================================================")
print("                  BENCHMARK SUMMARY")
print("=======================================================")
print(f"Metric             | deepCDG (Baseline)  | deepCDG-X (Proposed)")
print(f"-------------------|---------------------|---------------------")
print(f"ROC-AUC            | {np.mean(orig_auc_list):.4f} +/- {np.std(orig_auc_list):.4f}  | {np.mean(x_auc_list):.4f} +/- {np.std(x_auc_list):.4f}")
print(f"AUPRC              | {np.mean(orig_aupr_list):.4f} +/- {np.std(orig_aupr_list):.4f}  | {np.mean(x_aupr_list):.4f} +/- {np.std(x_aupr_list):.4f}")
print(f"Train Time (5 Folds)| {np.sum(orig_time_list):.2f}s               | {np.sum(x_time_list):.2f}s")
print(f"Uncertainty (Var)  | N/A                 | {np.mean(x_unc_list):.5f}")
print(f"Parameter Count    | {orig_params:,}             | {x_params:,}")
print(f"=======================================================")
