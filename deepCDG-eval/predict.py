import warnings
import copy
import numpy as np
import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipeline.sync.microbatch import scatter
from torch_geometric.utils import remove_self_loops, add_self_loops
from utils import get_ppi, k_folds, set_seed
from model import Net
import argparse
from tqdm import tqdm
from sklearn import metrics
from plot import plot
import json
from torch_geometric.nn import GNNExplainer
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
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

torch.autograd.set_detect_anomaly(True)

device = torch.device(args.device)
data = get_ppi(args.dataset, PATH='./PPI_data/')
data.x = data.x[:, :48]

num_nodes = data.y.shape[0]
all_idx = np.array(range(num_nodes))
num_features = data.x.shape[1]
num_classes = data.y.max().item() + 1

data = data.to(device)

all_mask = (data.train_mask | data.val_mask | data.test_mask).cpu().detach().numpy()
mask = [i for (i, x) in enumerate(all_mask) if data.y[i]]
model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in tqdm(range(1, args.epochs + 1)):
    # 训练模型
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    cls_loss = F.binary_cross_entropy_with_logits(pred[all_mask], data.y[all_mask].float())
    loss = cls_loss
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data)
all_gene = data.name

pred_gene = all_gene[~all_mask]
pred_score = torch.sigmoid(pred[~all_mask]).squeeze().cpu().detach().numpy()

# 保存结果为csv文件，将建一个路径为pred_result的文件夹，里面存放了结果文件
pred_result_path = 'pred_result'
if not os.path.exists(pred_result_path):
    os.makedirs(pred_result_path)
result = pd.DataFrame({'pred_gene': pred_gene, 'pred_score': pred_score})
# 对result按照pred_score进行排序：
result_sorted = result.sort_values(by='pred_score', ascending=False)
result_sorted.to_csv(f'{pred_result_path}/{args.dataset}_pred_result.csv', index=False)
