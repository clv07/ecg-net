# Heart Disease Classification on ECG

## Setup

1. Prepare the dataset by downloading [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) from PhysioNet and placing it at `./ptb-xl/`. The directory should contain `ptbxl_database.csv`, `scp_statements.csv`, and the `records100/` folder.

2. Install dependencies:
```bash
pip install torch wfdb scipy scikit-learn xgboost neurokit2 tqdm pandas pyarrow numpy
```

## Reproduce

### Baselines and CNN + Transformer Proposed Model
- Run the cells in `baseline-and-cnn-transformer.ipynb` from top to bottom. The notebook handles data loading, training, checkpoint selection, threshold tuning, and evaluation.

### Multi-Scale Per-Lead Attention Model
- Run `python ms_perlead_attn.py` end-to-end. The script handles data loading, training, checkpoint selection, threshold tuning, and evaluation automatically.

### ECGInceptionSENet
- Run ptbxl_ecg_classification.ipynb end-to-end on Google Colab (T4 GPU recommended). The notebook handles data loading, preprocessing, training, evaluation, and figure generation.

### L5G-Net
- Run `L5GNet_reproduction.ipynb` cell by cell. The script handles data loading, training, and evaluation.
