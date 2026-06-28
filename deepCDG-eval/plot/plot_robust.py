import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager


Dataset = ['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015']
Modelname = ['deepCDG', 'HGDC' , 'IMVRL-GCN', 'MTGCN', 'SMG', 'proEMOGI']
Perturbation = ['Rewired']
marker = ['v-', 's-', 'H-', 'd-', 'o-', 'D-']
Perturbation_ratio = ['0.0', '0.25', '0.5', '0.75', '0.9']
plt.figure(figsize=(3,6), dpi=600)
for (j, perturbation) in enumerate(Perturbation):
    for (i, model) in enumerate(Modelname):
        AUPRC = []
        directory = f'robust_{perturbation}'
        for perturbation_ratio in Perturbation_ratio:
            file_name = f"{Dataset}_{perturbation_ratio}.json"
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                if model == 'MTGCN' or model == 'proEMOGI':
                    AUPRC.append(data['test_auprc_mean'])
                else:
                    AUPRC.append(data['mean_auprc'])
        if model == 'proEMOGI':
            plt.plot([0.0, 0.25, 0.5, 0.75, 0.9], AUPRC, marker[i], label='EMOGI')
        else:
            plt.plot([0.0, 0.25, 0.5, 0.75, 0.9], AUPRC, marker[i], label=f'{model}')
        plt.xticks([0.0, 0.25, 0.5, 0.75, 0.9])
        plt.yticks([0.6, 0.65, 0.70, 0.75, 0.80, 0.85])
        plt.legend(loc='lower left')
    plt.xlabel('Perturbation')
    plt.ylabel('AUPRC')
    plt.title(f'{perturbation} robustness')
plt.savefig(f'robust.png', bbox_inches='tight')
plt.show()

