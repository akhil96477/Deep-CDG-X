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
from sklearn.model_selection import StratifiedKFold
from utils import get_ppi, k_folds, set_seed
from model import Net
import argparse
from tqdm import tqdm
from sklearn import metrics
from plot import plot
import random
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
parser.add_argument('--alpha', type=float, default=0.15)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--use_5_CV_pkl', type=bool, default=True,
                    help='whether to use existed 5_CV.pkl')
parser.add_argument('--logs', default=True, help='Save the results to a log file')
parser.add_argument('--times', type=int, default=1, help='Times of 5_CV')
parser.add_argument('--method_name', type=str, default='Mymodel')
parser.add_argument('--perturbation', type=str, default='Rewired', help='which type perturbation', choices=['Network','Feature', 'Rewired'])
parser.add_argument('--perturbation_ratio', type=float, default=0.)
args = parser.parse_args()

set_seed(args.seed)

def rewired_edges(edge_index, ratio=0.25):
    # 获取所有节点
    if ratio == 0:
        return edge_index
    all_nodes = torch.unique(edge_index)
    # 执行删除边的操作
    rew_num = int(edge_index.shape[1] * ratio)
    src = edge_index[0, :]
    dst = edge_index[1, :]
    rew_src = np.random.choice(len(src), rew_num, replace=False)
    rew_dst = np.random.choice(len(src), rew_num, replace=False)

    dst[rew_src] = dst[rew_dst]

    # 重构扰动后的边索引
    perturbed_edge_index = torch.stack([src, dst], dim=0)
    return perturbed_edge_index

def permute_edges(edge_index, ratio=0.25):
    # 获取所有节点
    all_nodes = torch.unique(edge_index)
    # 执行删除边的操作
    del_num = int(edge_index.shape[1] * ratio)
    src = edge_index[0, :]
    dst = edge_index[1, :]
    del_idx = np.random.choice(len(src), del_num, replace=False)
    perturbed_src = np.delete(src, del_idx)
    perturbed_dst = np.delete(dst, del_idx)

    # 重构扰动后的边索引
    perturbed_edge_index = torch.stack([perturbed_src, perturbed_dst], dim=0)

    # 检查并确保所有节点都出现在perturbed_edge_index中
    missing_nodes = torch.from_numpy(np.setdiff1d(all_nodes.numpy(), torch.unique(perturbed_edge_index).numpy()))
    if len(missing_nodes) > 0:
        # 处理缺失节点的操作，例如将缺失节点添加回perturbed_edge_index中
        for node in missing_nodes:
            perturbed_edge_index = torch.cat((perturbed_edge_index, torch.tensor([[node], [node]])), dim=1)

    # 最终的扰动后的边索引
    return perturbed_edge_index


def permute_features(x, ratio=0.25):
    # 生成随机索引，选择要替换的行
    random_indices = random.sample(range(x.size(0)), int(ratio * x.size(0)))
    # 生成随机数据，用于替换选定的行
    #random_data = torch.randn(len(random_indices), x.size(1))
    random_data = torch.zeros(len(random_indices), x.size(1))
    # 将选定的行替换为随机数据
    x[random_indices] = random_data

    return x

# load data
device = torch.device(args.device)
data = get_ppi(args.dataset, PATH='./PPI_data/')
data.x = data.x[:, :48]

if args.perturbation == 'Network':
    data.edge_index = permute_edges(data.edge_index, args.perturbation_ratio)

elif args.perturbation == 'Feature':
    data.x = permute_features(data.x, args.perturbation_ratio)

elif args.perturbation == 'Rewired':
    data.edge_index = rewired_edges(data.edge_index, args.perturbation_ratio)

data = data.detach().to(device)

# auc prc
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

@torch.no_grad()
def test(data, mask):
    model.eval()
    x = model(data.x, data.edge_index)
    pred = torch.sigmoid(x[mask])
    precision, recall, _ = metrics.precision_recall_curve(data.y[mask].cpu().numpy(),
                                                                    pred.cpu().detach().numpy())
    fpr, tpr, _ = metrics.roc_curve(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy())
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy()), area

kfold = 5

all_mask = (data.train_mask | data.val_mask | data.test_mask).cpu().numpy()
y = data.y.squeeze()[all_mask.squeeze()].cpu().numpy()
idx_list = np.arange(all_mask.shape[0])[all_mask.squeeze()]

skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)

train_mask_set = []
test_masks_set = []

# 根据五折交叉验证的划分结果生成train_mask和test_mask
for train_index, test_index in skf.split(idx_list, y): #划分训练集和测试集
    train_mask = np.full_like(all_mask, False)  # 初始化与all_mask相同大小的train_mask
    test_mask = np.full_like(all_mask, False)  # 初始化与all_mask相同大小的test_mask

    # 将训练集索引位置设置为True
    train_mask[idx_list[train_index]] = True
    # 将测试集索引位置设置为True
    test_mask[idx_list[test_index]] = True

    train_mask_set.append(train_mask)
    test_masks_set.append(test_mask)

# Ten times of 5_CV
best_test_auprc_list = []
best_test_auc_list = []

# train
for i in range(args.times):
    for train_mask, test_mask in zip(train_mask_set, test_masks_set):
        tr_mask, te_mask = train_mask, test_mask
        model = Net(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            cls_loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].float())
            #ce_loss = F.cross_entropy(predictor_out, ol.squeeze().long())
            loss = cls_loss #+ ce_loss * args.alpha
            loss.backward()
            optimizer.step()
        AUC, AUPR = test(data, te_mask)
        print(f'AUC: {AUC}, AUPR: {AUPR}')
        best_test_auc_list.append(AUC)
        best_test_auprc_list.append(AUPR)

mean_auc = np.mean(best_test_auc_list)
mean_auprc = np.mean(best_test_auprc_list)

std_auc = np.std(best_test_auc_list)
std_auprc = np.std(best_test_auprc_list)

print('Mean AUC: %.5f, Mean AUPR: %.5f' % (mean_auc, mean_auprc))
print('Std AUC: %.5f, Std AUPR: %.5f' % (std_auc, std_auprc))

if args.logs:
    # 将args以及mean_auc、mean_auprc、std_auc、std_auprc保存到json文件中
    args_dict = vars(args)
    args_dict['mean_auc'] = mean_auc
    args_dict['mean_auprc'] = mean_auprc
    args_dict['std_auc'] = std_auc
    args_dict['std_auprc'] = std_auprc

    import json
    import os

    directory = f'robust_{args.perturbation}'
    # Define the file path and name based on args.dataset and current time
    file_name = f"{args.dataset}_{args.perturbation_ratio}.json"
    file_path = os.path.join(directory, file_name)

    # Check if the metric_result directory exists, and create it if not
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the data dictionary to a YAML file
    with open(file_path, 'w') as file:
        json.dump(args_dict, file, indent=4)

    print(f"Saved the results to a log file: {file_path}")
