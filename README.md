# GMAMLink: GraphMAE and GCN-based Link Prediction for Gene Regulatory Network Inference

This repository contains the implementation of **GMAMLink**, a novel computational framework for inferring Gene Regulatory Networks (GRNs) from single-cell RNA sequencing (scRNA-seq) data using Graph Masked Autoencoders (GraphMAE) and Graph Convolutional Networks (GCN).

## Overview

GMAMLink leverages a two-stage learning process:
1. **Self-supervised Pre-training**: Utilizing GraphMAE to learn robust gene embeddings from expression data through masked feature reconstruction.
2. **Supervised Link Prediction**: Training a GCN-based LinkModel to predict regulatory interactions between Transcription Factors (TFs) and Target genes based on the learned embeddings.

## Project Structure

- `main.py`: The main entry point for running batch experiments across all datasets.
- `GraphMAE.py`: Implementation of the Graph Masked Autoencoder for feature learning.
- `Linkmodel.py`: Implementation of the link prediction model (GCN + Dot Product).
- `GCN.py`: Contains the GCN layers and dataset loading utilities.
- `utils.py`: Utility functions for normalization, evaluation (AUC/AUPR), and loss functions.
- `Benchmark_Dataset/`: Directory containing all datasets (STRING, Lofgof, Non-Specific, Specific).

## Getting Started

### 1. Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

The model expects the data to be organized in the `Benchmark_Dataset` directory with the following structure:

```
Benchmark_Dataset/
├── STRING_Dataset/      # Raw features (Expression, TFs, Targets)
├── STRING/              # Data splits (Train, Val, Test sets)
├── Specific_Dataset/
├── Specific/
├── Non-Specific_Dataset/
├── Non-Specific/
├── Lofgof_Dataset/
└── Lofgof/
```

### 3. Running Experiments

To reproduce the experiments described in the manuscript, simply run:

```bash
python main.py
```

The script will:
- Iterate through all 43+ benchmark experiments.
- Train the model on each dataset.
- Record the best AUC and AUPR achieved for each experiment.
- Track execution time and peak memory usage (RAM and GPU).

### 4. Results

After completion, the results will be saved in `results1.csv`. This file includes:
- **Experiment File**: Path to the dataset.
- **AUC**: Area Under the ROC Curve (Best achieved).
- **AUPR**: Area Under the Precision-Recall Curve (Best achieved).
- **Time (s)**: Training duration.
- **RAM Usage (MB)**: System memory consumption.
- **Peak GPU Memory (MB)**: Maximum GPU VRAM utilized.
