import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import os
import json
import math

# 变量名称
labels = ['deepCDG', 'HGDC' , 'IMVRL-GCN', 'MTGCN', 'SMG', 'EMOGI']
Dataset = ['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015']
Modelname = ['Mymodel', 'HGDC' , 'IMVRL-GCN', 'MTGCN', 'SMG', 'proEMOGI']

values = []

for (i, model) in enumerate(Modelname):
    AUPRC = []
    directory = f'time&cuda_result'
    for dataset in Dataset:
        file_name = f"{dataset}.json"
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
            AUPRC.append(math.log10(float(data['avg_epoch_time'])))
    values.append(AUPRC)
# 每个变量的值

# 为了使图形闭合，需要将第一个值重复到末尾
for value in values:
    value += value[:1]

# 计算角度
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 绘制雷达图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)

ax.fill(angles, values[0], color='#F27970', alpha=0.25)
ax.plot(angles, values[0], color='#F27970', linewidth=1, label=f'{labels[0]}')

ax.fill(angles, values[1], color='#BB9727', alpha=0.25)
ax.plot(angles, values[1], color='#BB9727', linewidth=1, label=f'{labels[1]}')

ax.fill(angles, values[2], color='#54B345', alpha=0.25)
ax.plot(angles, values[2], color='#54B345', linewidth=1, label=f'{labels[2]}')

ax.fill(angles, values[3], color='#C76DA2', alpha=0.25)
ax.plot(angles, values[3], color='#C76DA2', linewidth=1, label=f'{labels[3]}')

ax.fill(angles, values[4], color='#05B9E2', alpha=0.25)
ax.plot(angles, values[4], color='#05B9E2', linewidth=1, label=f'{labels[4]}')

ax.fill(angles, values[5], color='#8983BF', alpha=0.25)
ax.plot(angles, values[5], color='#8983BF', linewidth=1, label=f'{labels[5]}')

# 设置标签
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels([f'$\mathbf{{10^{i}}}$' for i in range(1, 5)])
ax.set_xticks(angles[:-1])
t = ax.set_xticklabels(Dataset, fontsize=10)
for i in t:
    i.set_position((0, -0.04))
# 设置标题和图例
plt.title('Model Running Time on Different Datasets', size=12, y=1.03)
plt.legend(loc='lower right', bbox_to_anchor=(0.14, -0.1), fontsize=9)
plt.savefig(f'time_result.png', dpi=600, bbox_inches='tight')
plt.show()