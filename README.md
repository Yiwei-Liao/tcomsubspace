# Sheaf Hypergraph Neural Networks

This repository contains the implementation of **Sheaf-enhanced Hypergraph Neural Networks**

## 🛠️ Installation

We recommend using **Conda** to manage the environment.

1. **Create and activate a new environment:**
   ```bash
   conda create -n sheaf_env python=3.10
   conda activate sheaf_env
   ```

2. **Install dependencies:**
   Ensure you have the `requirements.txt` file in the root directory.
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** Please ensure the directory structure is maintained for local imports (e.g., `models`, `layers`, `helper`, `sheaf_builder.py`, `sheafRHNN_conv.py`).

## 📂 Datasets

### 1. Link Prediction (Inference)
The datasets for link prediction tasks are pre-packaged in this repository under the `data/` folder:
- **FB-AUTO**: `data/FB-AUTO`
- **M-FB15K**: `data/M-FB15K`

### 2. Node Classification
For node classification tasks (e.g., Cora, PubMed, Citeseer), we utilize the preprocessing steps from the AllSet framework.

> **Setup Instructions:**
> To generate datasets, please follow the instructions from the [AllSet Repository](https://github.com/jianhao2016/AllSet). Once generated, place the `data` folder in the root directory of this project (e.g., `./data/`).

## 🚀 Quick Start

This repository provides two main entry scripts.

### Link Prediction (Inference)
Run the model for link prediction. 

```bash
python run_inference.py \
  -data data/FB-AUTO \
  -score_func conve \
  -opn corr \
  -gpu 0 \
  -epoch 5 \
  -batch 256
```

### Node Classification
Run the model for node classification.

```bash
python run_classification.py \
  --dname pubmed \
  --method sheafHyperGNNDiag \
  --cuda 0 \
  --epochs 10 \
  --heads 2 \
  --MLP_hidden 32
```

## 🧪 Experiments

We provide Shell scripts to reproduce the experiments regarding dimension variations and noise robustness (SNR tests).

**Prerequisite:** Ensure the scripts are executable:
```bash
chmod +x *.sh
```

### Available Experiments

1.  **Dimension Variation Test (Classification)**
    Runs the classification model across different hidden dimensions (2, 4, ..., 128).
    ```bash
    ./run_dim_test.sh
    ```

2.  **Noise Robustness Test (Classification)**
    Evaluates the model under Gaussian and Rayleigh noise with varying SNR (-6dB to 10dB).
    ```bash
    ./run_class_noise.sh
    ```

3.  **Noise Robustness Test (Inference)**
    Evaluates the link prediction model under different noise conditions.
    ```bash
    ./run_infer_noise.sh
    ```

## 📜 Citation

If you find this code useful for your research, please consider citing our work.
