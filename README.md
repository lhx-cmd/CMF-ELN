# CMF-ELN:  Cross-Modal-Fused End-to-end Learning Network for Cold-Start Drug-Drug Interaction Prediction

## Abstract

CMF-ELN is a deep learning framework for predicting drug-drug interactions (DDIs) using graph neural networks with multi-modal feature fusion. This repository contains the official implementation of our method, which integrates multiple heterogeneous drug feature sources through a graph convolutional network (GCN) architecture to achieve accurate DDI prediction and classification.

## Overview

Drug-drug interactions (DDIs) represent a critical challenge in pharmacology and clinical practice. This work presents a novel computational approach that leverages:

- **Multi-modal Feature Integration**: Fusion of four distinct drug feature modalities (chemical structure, biological activity, target information, etc.)
- **Graph Neural Network Architecture**: GCN-based encoders implemented with PyTorch Geometric
- **Multi-scenario Evaluation**: Support for Task 1 (S1 scenario) and Task 2 (S2 scenario) prediction tasks
- **Comprehensive Benchmarking**: Evaluation on two benchmark datasets (1,409 and 1,710 drugs)
- **Interpretability Analysis**: Feature visualization and attention weight analysis tools
- **Embedding Visualization**: t-SNE-based clustering analysis of learned drug representations

## Repository Structure

```
.
├── code/                          # Source code directory
│   ├── 1409/                      # Code for 1409-drug dataset
│   │   ├── modeltask2.py          # Model definition for Task 2
│   │   ├── modeltask3.py          # Model definition for Task 3
│   │   ├── task2.py               # Training script for Task 2 (S1 scenario)
│   │   └── task3.py               # Training script for Task 3 (S2 scenario)
│   ├── 1710/                      # Code for 1710-drug dataset
│   │   ├── modeltask2.py
│   │   ├── modeltask3.py
│   │   ├── task2.py
│   │   └── task3.py
│   ├── case_study/                # Case study analysis
│   │   ├── model_case.py          # Case study model
│   │   ├── save_weights.py        # Weight extraction utility
│   │   ├── modal2feature_case1.ipynb
│   │   ├── modal2feature_case2.ipynb
│   │   └── figures/               # Case study visualizations
│   └── cluster_analysis/          # Clustering analysis
│       ├── model_cluster.py       # Clustering model
│       ├── save_embedding.py      # Embedding extraction utility
│       ├── viusualize.ipynb       # Visualization notebook
│       └── figures/               # t-SNE visualization results
├── datasets/                      # Dataset directory
│   ├── 1409/                      # 1409-drug benchmark dataset
│   │   ├── DB1409.txt             # Drug identifier list
│   │   ├── DB_DDI.csv             # DDI labels
│   │   ├── DB_SMI.csv             # SMILES representations
│   │   ├── dataset1.txt           # Feature modality 1
│   │   ├── dataset2.txt           # Feature modality 2
│   │   ├── dataset3.txt           # Feature modality 3
│   │   ├── dataset4.txt           # Feature modality 4
│   │   └── process.py             # Data preprocessing script
│   ├── 1710/                      # 1710-drug benchmark dataset
│   ├── case_study/                # Case study data
│   └── cluster_analysis/          # Clustering analysis data
└── result/                        # Experimental results
    ├── 1409/
    └── 1710/
```

## Software Dependencies

The package development version is tested on _Linux_ (Ubuntu 20.04) operating systems with CUDA 12.4.

CMF-ELN is tested under ``Python == 3.10.14``.

To reproduce the experimental environment, the following Python packages are required:
```bash
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
torch-geometric==2.6.1
torch_scatter==2.1.2
numpy==2.2.5
pandas==2.2.0rc0
scikit-learn==1.7.2
scipy==1.15.2
umap-learn==0.5.9.post2
networkx==3.4.2
matplotlib==3.10.3
seaborn==0.13.2
pillow==10.3.0
tqdm==4.67.1
```

## Usage

### Task 1: S1 Scenario (Single-side Novel Drug)

**Prediction Scenario**: At least one drug in the pair has been observed during training.

```bash
# For 1409-drug dataset
cd code/1409
python task1.py

# For 1710-drug dataset
cd code/1710
python task2.py
```

### Task 2: S2 Scenario (Both-side Novel Drugs)

**Prediction Scenario**: Both drugs in the pair are unseen during training.

```bash
# For 1409-drug dataset
cd code/1409
python task1.py

# For 1710-drug dataset
cd code/1710
python task2.py
```
### Case Studies

The repository includes two case studies for analyzing the model's feature learning capabilities:

```bash
cd code/case_study
python save_weights.py
jupyter notebook modal2feature_case1.ipynb
jupyter notebook modal2feature_case2.ipynb
```

### Clustering Analysis

Visualize learned drug embeddings using t-SNE:

```bash
cd code/cluster_analysis
python save_embedding.py  # Extract embeddings
jupyter notebook viusualize.ipynb  # Generate visualizations
```
## Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@article{cmf-eln2026,
  title={CMF-ELN: Chemical Multi-Feature Enhanced Learning Network for Drug-Drug Interaction Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2026},
  volume={[Volume]},
  pages={[Pages]},
  doi={[DOI]}
}
```


