# deepCDG-eval
This repository contains the evaluation of deepCDG and the code which produces the plots in the manuscript.

To get the dataset, please follow the instructions from [https://github.com/xingyili/deepCDG](https://github.com/xingyili/deepCDG)
##  Robustness evaluation on pan-cancer datasets
Performance evaluation can be run by calling

``python robust.py --perburbation ["Network", "Feature", "Rewired"] -- perburbation_ratio [0, 0.25, 0.5, 0.75, 0.9]``

## Time overhead analysis
Time overhead analysis can be run by calling

``python time&cuda.py``

## Evaluation on cancer type-specific driver gene prediction
Performance evaluation can be run by calling

``python specific.py``

## Performance evaluation on independent test sets
Performance evaluation on independent test can be run by calling

``python independent_dataset.py``

## Prediction of potential cancer driver genes by deepCDG
Prediction evaluation can be run bt calling

``python predict.py``

The prediction result will be saved as a csv profile. 

## Enrichment analysis
Since you can get the prediction of potential cancer driver genes list, you can do enrichment analysis on these genes using any tools such as R.

## Drug sensitivity analysis
You can visit the [Gene Set Cancer analysis](http://bioinfo.life.hust.edu.cn/GSCA) to do drug sensitivity analysis using the prediction of cancer driver gene.

## Gene module dissection in pan-cancer
Gene module dissection can be run by calling

``python gene_module.py``
