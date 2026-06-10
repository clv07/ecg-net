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