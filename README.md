# Heart Disease Classification on ECG

## Setup

1. Prepare the dataset by downloading [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) from PhysioNet and placing it at `./ptb-xl/`. The directory should contain `ptbxl_database.csv`, `scp_statements.csv`, and the `records100/` folder.

2. Install dependencies:
```bash
pip install torch wfdb scipy scikit-learn xgboost neurokit2 tqdm pandas pyarrow numpy
```

## Reproduce

### Baselines and CNN + Transformer Proposed Model
- Run the cells in `baseline-and-cnn-transformer.ipynb` from top to bottom.

### ECGInceptionSENet
- ECGInceptionSENet is a multi-label ECG classifier trained on the PTB-XL dataset to detect five cardiac diagnostic superclasses (NORM, MI, STTC, CD, HYP) from standard 12-lead recordings.
- The architecture enhances InceptionTime with Squeeze-and-Excitation modules for adaptive channel recalibration, multi-head self-attention for long-range temporal reasoning, and Asymmetric Loss to handle class imbalance.
- On the held-out test fold, the model achieves Macro AUROC 0.919, Macro AUPRC 0.803, and an optimised Macro F1 of 0.735 after per-class threshold tuning.
- To reproduce results, run ptbxl_ecg_classification.ipynb end-to-end on Google Colab (T4 GPU recommended); the notebook handles data loading, preprocessing, training, evaluation, and figure generation automatically.

### MS Per-Lead Attn
- MS Per-Lead Attn is a multi-label ECG classifier trained on the PTB-XL dataset to detect five cardiac diagnostic superclasses (NORM, MI, STTC, CD, HYP) from standard 12-lead recordings.
- The architecture extends the per-lead CNN and spatial lead attention framework with a multi-scale grouped convolution frontend (kernel sizes 5, 15, and 31), followed by cross-lead self-attention at each time step and a temporal Transformer for long-range sequence modeling.
- On the held-out test fold, the model achieves Macro AUROC 0.917, Macro Precision 0.701, and an optimised Macro F1 of 0.737 after per-class threshold tuning.
- To reproduce results, run `python preprocess.py` once (if `./preprocessed/` is not already available), then run `python MS_Per-Lead_Attn.py` end-to-end; the script handles data loading, training, checkpoint selection, threshold tuning, and evaluation automatically (CUDA, Apple MPS, or CPU via `device.py`).

### L5G-Net
- L5G-Net is a multi-label ECG classifier trained on the PTB-XL dataset to detect five cardiac diagnostic superclasses (NORM, MI, STTC, CD, HYP) from standard 12-lead recordings.
- L5G-Net introduce medical perspective into the machine learning, combine lead into groups, also modify CNN model to accommodate with the group training.
- On the held-out test fold, the model achieves Macro AUROC 0.940, Precision 0.8743, and an Macro F1 of 0.753.
- To reproduce results, run `L5GNet_reproduction.ipynb` cell by cell, the script handles data loading, training, and evaluation automatically.
