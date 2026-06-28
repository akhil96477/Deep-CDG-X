import os
import pandas as pd
import csv
from utils import get_ppi, k_folds, set_seed
import numpy as np
import torch
from sklearn import metrics
import argparse
from model import Net
from tqdm import tqdm
import json
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CPDB',
                    choices=['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015'],
                    help="The dataset to be used.")
parser.add_argument('--device', type=str, default='cuda:0',
                    choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--hidden_channel_1', type=int, default=48)
parser.add_argument('--hidden_channel_2', type=int, default=200)
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--alpha', type=float, default=0.15)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_5_CV_pkl', type=bool, default=True,
                    help='whether to use existed 5_CV.pkl')
parser.add_argument('--logs', default=False, help='Save the results to a log file')
parser.add_argument('--times', type=int, default=1, help='Times of 5_CV')
parser.add_argument('--method_name', type=str, default='Mymodel')
import torch.nn.functional as F
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(args.device)

path = f'./data/Benchmark_NCG61.csv'
df = pd.read_csv(path)

data = get_ppi(args.dataset, PATH='./PPI_data/')
length = len(data.name)

cancer = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD',
          'HNSC', 'LUAD', 'THCA', 'BLCA', 'ESCA',
          'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

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

sx = data.x
args_dict = {}
for (i, cancer) in enumerate(cancer):
    if cancer == 'READ':
        continue
    print(cancer)
    gene = set([f'{string}' for (id, string) in enumerate(df['symbol']) if df['Cohort'][id] == cancer])
    y = torch.zeros(length, 1, dtype=torch.int32)
    for (id, name) in enumerate(data.name):
        if name in gene:
            y[id][0] = 1
    data.y = y
    data.x = sx[:, (i, i + 16, i + 32)]
    data = data.to(device)
    k_sets = k_folds(data)

    AUC = np.zeros(shape=(1, 5))
    AUPR = np.zeros(shape=(1, 5))
    # train
    auprc = 0
    cnt = 0
    for j in range(args.times):
        for cv_run in range(5):
            tr_mask, te_mask = k_sets[j][cv_run]
            model = Net(args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            for epoch in tqdm(range(args.epochs)):
                model.train()
                optimizer.zero_grad()
                pred = model(data.x, data.edge_index)
                cls_loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].float())
                # ce_loss = F.cross_entropy(predictor_out, ol.squeeze().long())
                loss = cls_loss  # + ce_loss * args.alpha
                loss.backward()
                optimizer.step()
            AUC[j][cv_run], AUPR[j][cv_run] = test(data, te_mask)
            auprc += AUPR[j][cv_run]
            cnt += 1
            print(f'AUC: {AUC[j][cv_run]}, AUPR: {AUPR[j][cv_run]}, cv_run: {cv_run}, mean: {auprc / cnt}')
        print(f'AUC: {np.mean(AUC[j])}', f'AUPR: {np.mean(AUPR[j])}')
        args_dict[f'{cancer}_AUC'] = np.mean(AUC[j])
        args_dict[f'{cancer}_AUPRC'] = np.mean(AUPR[j])

if args.logs:
    directory = f'type-specific'
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
