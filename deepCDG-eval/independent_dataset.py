#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/5/8 10:55
# @Author  : Jimmy
# @FileName: independent_set.py
# @Usage: say something
import sys, argparse
from model import *
from utils import *
import json
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='STRINGdb',
                    choices=['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015'],
                    help="The dataset to be used.")
parser.add_argument('--device', type=str, default='cuda:0',
                    choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
parser.add_argument('--in_channels', type=int, default=16)
parser.add_argument('--hidden_channel_1', type=int, default=48)
parser.add_argument('--hidden_channel_2', type=int, default=200)
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--alpha', type=float, default=0.15)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_5_CV_pkl', type=bool, default=True,
                    help='whether to use existed 5_CV.pkl')
parser.add_argument('--logs', default=False, help='Save the results to a log file')
parser.add_argument('--times', type=int, default=1, help='Times of 5_CV')
parser.add_argument('--method_name', type=str, default='Mymodel')
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(args.device)
torch.autograd.set_detect_anomaly(True)

def load_ongene(PATH='./data/ongene_human.txt'):
    ongene = pd.read_csv(PATH, sep='\t')
    gene_list = ongene['OncogeneName'].tolist()
    return gene_list

def load_oncokb(PATH='./data/OncoKB_cancerGeneList.tsv'):
    oncokb = pd.read_csv(PATH, sep='\t')
    gene_list = oncokb['Hugo Symbol'].tolist()
    return gene_list

""" Experimental settings """
data = get_ppi(dataset=args.dataset, PATH='./PPI_data/')
data.x = data.x[:, :48]
num_nodes = data.y.shape[0]
all_idx = np.array(range(num_nodes))
num_features = data.x.shape[1]
num_classes = data.y.max().item() + 1

data = data.to(device)
model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
all_mask = (data.train_mask | data.val_mask | data.test_mask).cpu().detach().numpy()

ongene = load_ongene()
oncokbgene = load_oncokb()
unlabel_gene = data.name[~all_mask]
on_TP_gene = set(ongene) & set(unlabel_gene)
oncokb_TP_gene = set(oncokbgene) & set(unlabel_gene)

on_persudo_true = [1 if unlabel_gene[i] in on_TP_gene else 0 for i in range(len(unlabel_gene))]
oncokb_persudo_true = [1 if unlabel_gene[i] in oncokb_TP_gene else 0 for i in range(len(unlabel_gene))]

on_best_auprc = 0
on_best_epoch = None

oncokb_best_auprc = 0
oncokb_best_epoch = None

for epoch in tqdm(range(1, args.epochs + 1)):
    # 训练模型
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.binary_cross_entropy_with_logits(out[all_mask], data.y[all_mask].float())
    loss.backward()
    optimizer.step()

    model.eval()
    pred = model(data.x, data.edge_index)

    pred_score = torch.sigmoid(pred[~all_mask]).squeeze().cpu().detach().numpy()
    on_auprc = compute_auprc(on_persudo_true, pred_score)
    oncokb_auprc = compute_auprc(oncokb_persudo_true, pred_score)

    if on_auprc > on_best_auprc:
        on_best_auprc = on_auprc
        on_best_epoch = epoch

    if oncokb_auprc > oncokb_best_auprc:
        oncokb_best_auprc = oncokb_auprc
        oncokb_best_epoch = epoch

print("Best AUPRC oncogene: {:.4f} at epoch {}".format(on_best_auprc, on_best_epoch))
print("Best AUPRC oncokb: {:.4f} at epoch {}".format(oncokb_best_auprc, oncokb_best_epoch))
print("oncogene: {:.4f}".format(on_auprc))
print("oncokb: {:.4f}".format(oncokb_auprc))


if args.logs:
    args_dict = {}
    args_dict["AUPRC oncogene"]=on_best_auprc
    args_dict["Bset_epoch oncogene"]=on_best_epoch
    args_dict["AUPRC oncokb"]=oncokb_best_auprc
    args_dict["Bset_epoch oncokb"]=oncokb_best_epoch

    directory = f'independent_set'
    # Define the file path and name based on args.dataset and current time
    file_name = f"{args.dataset}.json"
    file_path = os.path.join(directory, file_name)

    # Check if the metric_result directory exists, and create it if not
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the data dictionary to a YAML file
    with open(file_path, 'w') as file:
        json.dump(args_dict, file, indent=4)

    print(f"Saved the results to a log file: {file_path}")


