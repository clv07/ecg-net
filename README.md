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
