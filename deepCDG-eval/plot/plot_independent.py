import numpy as np
import os
import json
import matplotlib.pyplot as plt


Dataset = ['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015']
Modelname = ['deepCDG', 'HGDC' , 'IMVRL-GCN', 'MTGCN', 'SMG', 'proEMOGI']
Perturbation = ['Network', 'Feature']
marker = ['v', 's', 'H', 'd', 'o', 'D']
color = ['b', 'g', 'r', 'c', 'm', 'y']

plt.figure()
handle1 = []
handle2 = []

for (i, dataset) in enumerate(Dataset):
    for (j, model) in enumerate(Modelname):
        directory = f'independent_set'
        file_name = f"{dataset}.json" // 保存的结果json文件
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
            point1, = plt.plot(data['AUPRC oncogene'], data['AUPRC oncokb'], marker[i], color=
                               'gray', label=f'{dataset}')
    handle1.append(point1)
leg1 = plt.legend(handles=handle1, loc='lower right', fontsize=8, title='dataset')
handles, labels = plt.gca().get_legend_handles_labels()

for (i, handle) in enumerate(handles):
    handle.set_color(color[i % 6])
    handle.set_alpha(0.2)

for (i, model) in enumerate(Modelname):
    ongene_AUPRC = []
    oncokb_AUPRC = []
    for (j, dataset) in enumerate(Dataset):
        directory = f'independent_set'
        file_name = f"{dataset}.json"
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
            ongene_AUPRC.append(data['AUPRC oncogene'])
            oncokb_AUPRC.append(data['AUPRC oncokb'])
    if model == 'proEMOGI':
        point2, = plt.plot(np.mean(ongene_AUPRC), np.mean(oncokb_AUPRC), 'o', color=color[i], markersize=10,
                           label=f'EMOGI')
    else:
        point2, = plt.plot(np.mean(ongene_AUPRC), np.mean(oncokb_AUPRC), 'o', color=color[i], markersize=10, label=f'{model}')
    handle2.append(point2)
plt.legend(handles=handle2, loc='upper left')
plt.gca().add_artist(leg1)
plt.xlabel('AUPRC for oncogenes(ONGene)')
plt.ylabel('AUPRC for oncoKB high confidence')
plt.title('')
plt.savefig(f'independent_set.pdf')
plt.show()

