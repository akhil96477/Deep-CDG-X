#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/5/3 16:00
# @Author  : Jimmy
# @FileName: predict.py
# @Usage: say something
import argparse
import numpy as np
import torch
from sklearn import metrics
from model import Net
import torch.nn.functional as F
from benchmark.HGDC.utils.auxiliary_graph_generator import generate_auxiliary_graph
from benchmark.HGDC.utils.mytools import compute_auc, compute_auprc, get_ppi, set_seed
from sklearn.model_selection import StratifiedKFold
import os
import time
from datetime import datetime
import json

# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CPDB',
                    choices=['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015'],
                    help="The dataset to be used.")
parser.add_argument('--device', type=str, default='cuda:2',
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
parser.add_argument('--use_5_CV_pkl', type=bool, default=True,
                    help='whether to use existed 5_CV.pkl')
parser.add_argument('--logs', default=True, help='Save the results to a log file')
parser.add_argument('--times', type=int, default=1, help='Times of 5_CV')
parser.add_argument('--method_name', type=str, default='Mymodel')
args = parser.parse_args()

set_seed(args.seed)

device = torch.device(args.device)

# data = load_net_specific_data(args)
data = get_ppi(args.dataset, PATH='./PPI_data/')
data.x = data.x[:, :48]
data = data.to(device)

all_mask = (data.train_mask | data.val_mask | data.test_mask).cpu().numpy()
y = data.y.squeeze()[all_mask.squeeze()].cpu().numpy()
idx_list = np.arange(all_mask.shape[0])[all_mask.squeeze()]
model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Training loop
for epoch in range(1, args.epochs + 1):
    # Training model
    model.train()
    optimizer.zero_grad()
    pred, predictor_out = model(data)
    cls_loss = F.binary_cross_entropy_with_logits(pred[all_mask], data.y[all_mask].float())
    loss = cls_loss
    loss.backward()
    optimizer.step()

# End time
end_time = time.time()

# Total duration
total_duration = end_time - start_time

# GPU memory usage
if torch.cuda.is_available():
    memory_allocated = torch.cuda.max_memory_allocated(device)
    memory_reserved = torch.cuda.max_memory_reserved(device)
else:
    memory_allocated = 0
    memory_reserved = 0

print(f'Total training time: {total_duration:.2f} seconds.')
print(f'Memory Allocated: {memory_allocated / (1024 ** 2):.2f} MB, '
      f'Memory Reserved: {memory_reserved / (1024 ** 2):.2f} MB')


# 保留avg_epoch_time四位小数，avg_gpu_mem_usage转换为MB
total_duration = round(total_duration, 4)
memory_allocated = round(memory_allocated / (1024 ** 2), 2)

if args.logs:
    directory = "time&cuda_result"

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path and name based on args.dataset and current time
    # file_name = f"{args.dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
    file_name = f"{args.dataset}.json"
    file_path = os.path.join(directory, file_name)

    # Check if the metric_result directory exists, and create it if not
    if not os.path.exists('time&cuda_result'):
        os.makedirs('time&cuda_result')

    write_log = {
        'avg_epoch_time': total_duration,
        'avg_gpu_mem_usage': f"{memory_allocated} MB",
    }

    # Save the data dictionary to a YAML file
    with open(file_path, 'w') as file:
        json.dump(write_log, file, indent=4)



