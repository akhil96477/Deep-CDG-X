import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager


Dataset = ['CPDB', 'STRINGdb', 'MULTINET', 'PCNet', 'IRefIndex', 'IRefIndex_2015']
for i in range(len(Dataset)):
    dataset = Dataset[i]
    Modelname = ['deepCDG', 'HGDC' , 'MTGCN', 'IMVRL-GCN', 'EMOGI', 'SMG']
    for model in Modelname:
        if os.path.exists(f'/{dataset}/{model}_mean_recall.npy'):
            mean_recall = np.load(f'/{dataset}/{model}_mean_recall.npy')
            mean_precision = np.load(f'{dataset}/{model}_mean_precision.npy')

            plt.plot(mean_recall, mean_precision, label=f'{model}(AUPRC = {np.trapz(mean_precision, mean_recall):.4f})')
            plt.legend(loc='lower left')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR-curve for {dataset}')
    plt.savefig(f'PR-curve for {dataset}.png', bbox_inches='tight', dpi=600)
    plt.show()
