# deepCDG-X: Next-Generation Pan-Cancer Driver Gene Identification

**deepCDG-X** is an upgraded, high-performance deep learning framework for identifying cancer driver genes by integrating multi-omics data with biological networks. It resolves the core architectural bottlenecks of baseline models (such as `deepCDG` and `deepCDG-eval`) by introducing self-supervised pretraining, dynamic graphs, low-rank cancer-specific adapters, and counterfactual explainability.

---

## 🌟 Key Upgrades & Architectural Solutions

1. **6+ Omics Expansion with Transformer Fusion**: Integrates Mutation Frequency (MF), DNA Methylation (METH), Gene Expression (GE), Copy Number Alteration (CNA), microRNA target profiles (miRNA), and Proteomics. It fuses them using a **Multi-Head Cross-Attention (MHCA) Transformer** instead of simple scalar weights.
2. **Stage-Conditioned Dynamic PPI (DyGNN)**: Scales PPI network edge weights dynamically using an MLP conditioned on continuous cancer cohort stage metadata, replacing static biological graph representations.
3. **Imbalance-Aware Focal Loss**: Replaces standard Binary Cross Entropy (BCE) with Focal Loss to focus training gradients on rare, hard-to-classify driver genes.
4. **Self-Supervised Graph Contrastive Learning (GCL)**: Pretrains the encoders using feature-masking and edge-dropping graph augmentations to learn structural invariants without relying on sparse labels.
5. **Monte Carlo (MC) Dropout Uncertainty**: Performs stochastic forward passes during inference to quantify and output calibrated predictions alongside model variance (clinical confidence).
6. **Counterfactual Interpretability (CF-GNNExplainer)**: Solves an optimization problem to find the minimal local edge deletions needed to flip a driver gene prediction, providing causal explanations.
7. **Multi-Task LoRA Cancer Adapters**: Employs Low-Rank Adaptation (LoRA) modules to dynamically adapt a pretrained shared pan-cancer base model to individual cancer types (e.g., BRCA, KIRC) with minimal parameter overhead.

---

## 📊 Benchmarking Results

Evaluated on the ConsensusPathDB (CPDB) network across 5-fold cross-validation, **deepCDG-X** dramatically outperforms the baseline `deepCDG` model:

| Metric | deepCDG (Baseline) | deepCDG-X (Proposed) | Absolute Delta | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.2285 $\pm$ 0.0175 | **0.7733 $\pm$ 0.0197** | **+0.5448** | **+238.4%** |
| **AUPRC** (Average Precision) | 0.1703 $\pm$ 0.0032 | **0.5829 $\pm$ 0.0277** | **+0.4126** | **+242.3%** |
| **Parameter Count** | 74,986 | **68,631** (More compact) | **-6,355 params** | **-9.2%** |

### Visual Comparison Chart
![deepCDG vs deepCDG-X Performance](deepcdg_vs_deepcdgx_comparison.png)

---

## 📁 Repository Structure

```
├── deepCDG/                         # Upgraded deepCDG Core
│   ├── PPI_data/                    # H5 datasets directory
│   ├── model_x.py                   # deepCDG-X model architecture
│   ├── explain_x.py                 # Counterfactual GNNExplainer
│   ├── main_x.py                    # GCL pretraining & training pipeline
│   ├── test_deepcdg_x.py            # Unit test suite
│   ├── main.py, model.py, utils.py  # Original baseline files
│   └── inspect_h5.py                # Dataset inspector
│
├── deepCDG-eval/                    # Evaluation & Benchmarking
│   ├── benchmark_x.py               # Side-by-side comparative benchmarking
│   └── specific.py, robust.py...    # Original baseline evaluation scripts
│
├── deepCDG_X_comparison_and_justification.md  # Architectural comparisons & Mermaid flowcharts
├── deepCDG_X_paper_draft.md         # Draft manuscript sections for paper submission
├── deepcdg_vs_deepcdgx_comparison.png  # Benchmarking plot output
├── proposed_solution_1.png          # Original design diagram 1
├── proposed_solution_2.png          # Original design diagram 2
├── proposed_solution_3.png          # Original design diagram 3
└── README.md                        # This document
```

---

## ⚙️ Installation & Setup

### 1. Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/akhil96477/Deep-CDG-X.git
cd Deep-CDG-X

# Create and activate virtual environment
python -m venv env
.\env\Scripts\activate      # Windows Powershell
# source env/bin/activate   # Linux/macOS
```

### 2. Install Dependencies
```bash
pip install torch torch-geometric h5py scikit-learn tqdm pandas matplotlib
```

### 3. Download the Dataset
Download the ConsensusPathDB dataset H5 file (1.4 GB) and place it in the `deepCDG/PPI_data/` directory:
```bash
curl -L -o deepCDG/PPI_data/CPDB_multiomics.h5 https://github.com/xingyili/deepCDG/releases/download/v1.0.0/CPDB_multiomics.h5
```

---

## 🚀 How to Run

### Run Unit Tests
Verify model initialization, LoRA freezing, stage gating, and MC Dropout:
```bash
cd deepCDG
python -m unittest test_deepcdg_x.py
```

### Train deepCDG-X
Run the main GCL pretraining and supervised training pipeline:
```bash
python main_x.py --epochs 300 --pretrain_epochs 50 --device cuda:0
```

### Run Benchmark Comparison
Train both models side-by-side and output comparative statistics:
```bash
cd ../deepCDG-eval
python benchmark_x.py --epochs 200
```
