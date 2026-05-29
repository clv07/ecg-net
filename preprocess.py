"""
PTB-XL Unified Preprocessing Pipeline
======================================
Produces a single canonical version of the preprocessed dataset used by
ALL four models in the ablation study:

    1. XGBoost (with hand-crafted features built on top)
    2. Vanilla 1D CNN
    3. ResNet1D
    4. Pure Transformer (patch-embedding)
    5. CNN + Transformer (our proposed)

Outputs are saved to ./preprocessed/ as .npy files so training scripts
load instantly instead of re-reading 21k WFDB records every run.

Run once:
    python preprocess.py

Then in any training script:
    from preprocess import load_preprocessed
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed()
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR        = "./ptb-xl/"
OUT_DIR         = "./preprocessed/"
SAMPLING_RATE   = 100
SUPERCLASSES    = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES      = ["I","II","III","aVR","aVL","aVF",
                   "V1","V2","V3","V4","V5","V6"]

# Bandpass filter cutoffs
BANDPASS_LOW    = 0.5   # Hz — removes baseline wander (respiratory drift)
BANDPASS_HIGH   = 40    # Hz — removes muscle/HF noise
BANDPASS_ORDER  = 4

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# STEP 1 — Load metadata and aggregate diagnostic superclass
# ============================================================
def load_metadata(data_dir):
    print("[1/5] Loading metadata...")
    df = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"))
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    agg = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    agg = agg[agg.diagnostic == 1]

    def aggregate(y_dic):
        return list({agg.loc[k].diagnostic_class
                     for k in y_dic if k in agg.index})

    df["superclass"] = df.scp_codes.apply(aggregate)
    df["has_valid"]  = df.superclass.apply(
        lambda y: any(c in SUPERCLASSES for c in y)
    )
    return df


# ============================================================
# STEP 2 — Load raw WFDB signals
# ============================================================
def load_signals(df, data_dir, sampling_rate):
    print("[2/5] Loading raw WFDB signals (this takes a few minutes)...")
    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr
    signals = []
    for f in tqdm(filenames, desc="WFDB"):
        sig, _ = wfdb.rdsamp(os.path.join(data_dir, f))
        signals.append(sig.T)   # (12, T)
    return np.array(signals, dtype=np.float32)


# ============================================================
# STEP 3 — Bandpass filter
# ============================================================
def bandpass_filter(signals, low, high, fs, order):
    """
    Zero-phase Butterworth bandpass filter.
    signals: (N, 12, T)  →  filtered (N, 12, T)
    """
    print(f"[3/5] Bandpass filtering {low}-{high} Hz...")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    # process in chunks to keep memory reasonable
    out = np.empty_like(signals)
    for i in tqdm(range(len(signals)), desc="Filter"):
        out[i] = sosfiltfilt(sos, signals[i], axis=-1).astype(np.float32)
    return out


# ============================================================
# STEP 4 — Train/val/test split using PTB-XL recommended folds
# ============================================================
def split_data(X, y, df):
    """
    PTB-XL recommended split: folds 1–8 train, 9 val, 10 test.
    Returns indices so we can also split the dataframe (for hand-crafted
    feature extraction in XGBoost baseline).
    """
    print("[4/5] Splitting train/val/test...")
    train_mask = (df.strat_fold <= 8).values
    val_mask   = (df.strat_fold == 9).values
    test_mask  = (df.strat_fold == 10).values

    splits = {
        "train": (X[train_mask], y[train_mask], df[train_mask].reset_index(drop=True)),
        "val":   (X[val_mask],   y[val_mask],   df[val_mask].reset_index(drop=True)),
        "test":  (X[test_mask],  y[test_mask],  df[test_mask].reset_index(drop=True)),
    }
    for name, (Xs, ys, _) in splits.items():
        print(f"   {name:5s}: X {Xs.shape}, y {ys.shape}, pos rate {ys.mean(0).round(3)}")
    return splits


# ============================================================
# STEP 5 — Per-lead z-score normalization using TRAIN statistics
# ============================================================
def normalize(splits):
    """
    Compute mean/std from training data only, then apply to all splits.
    This preserves absolute amplitude information per lead, which matters
    for HYP (hypertrophy diagnosis depends on R-wave voltage).
    """
    print("[5/5] Per-lead z-score normalization (using train stats)...")
    X_train = splits["train"][0]
    # shape: (N, 12, T)  →  reduce over N and T  →  (1, 12, 1)
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2), keepdims=True)
    print(f"   per-lead mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"   per-lead std  range: [{std.min():.4f}, {std.max():.4f}]")

    out = {}
    for name, (X, y, df_split) in splits.items():
        X_norm = ((X - mean) / (std + 1e-8)).astype(np.float32)
        out[name] = (X_norm, y, df_split)

    # save stats too (useful for inference on new data later)
    np.save(os.path.join(OUT_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(OUT_DIR, "norm_std.npy"),  std)
    return out


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("PTB-XL Unified Preprocessing Pipeline")
    print("=" * 60)

    # Skip if already cached
    if all(os.path.exists(os.path.join(OUT_DIR, f"{s}_X.npy"))
           for s in ["train", "val", "test"]):
        print("Preprocessed files already exist in", OUT_DIR)
        print("Delete them to regenerate. Exiting.")
        return

    # 1. metadata
    df = load_metadata(DATA_DIR)
    print(f"   total records : {len(df):,}")
    print(f"   valid records : {df.has_valid.sum():,}")

    # 2. signals
    signals = load_signals(df, DATA_DIR, SAMPLING_RATE)
    print(f"   raw shape     : {signals.shape}")

    # filter to valid records
    mask = df.has_valid.values
    signals = signals[mask]
    df = df[mask].reset_index(drop=True)

    # encode labels
    mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
    y = mlb.fit_transform(df.superclass).astype(np.float32)

    # 3. bandpass filter
    signals = bandpass_filter(signals, BANDPASS_LOW, BANDPASS_HIGH,
                              SAMPLING_RATE, BANDPASS_ORDER)

    # 4. split
    splits = split_data(signals, y, df)

    # 5. normalize
    splits = normalize(splits)

    # save .npy + dataframe metadata
    print("\nSaving preprocessed arrays to", OUT_DIR)
    for name, (X, y_split, df_split) in splits.items():
        np.save(os.path.join(OUT_DIR, f"{name}_X.npy"), X)
        np.save(os.path.join(OUT_DIR, f"{name}_y.npy"), y_split)
        df_split.to_pickle(os.path.join(OUT_DIR, f"{name}_df.pkl"))
        print(f"   {name}: X.npy ({X.nbytes / 1e6:.1f} MB), "
              f"y.npy, df.pkl")

    # save class info
    np.save(os.path.join(OUT_DIR, "classes.npy"), np.array(SUPERCLASSES))
    print("\nDone.")


# ============================================================
# LOADER (used by training scripts)
# ============================================================
def load_preprocessed(out_dir=OUT_DIR, with_df=False):
    """
    Load preprocessed splits. Use this in every training script.

    Returns
    -------
    if with_df=False:
        X_train, y_train, X_val, y_val, X_test, y_test
    if with_df=True:
        same as above + (df_train, df_val, df_test)
    """
    splits = {}
    for name in ["train", "val", "test"]:
        X = np.load(os.path.join(out_dir, f"{name}_X.npy"))
        y = np.load(os.path.join(out_dir, f"{name}_y.npy"))
        splits[name] = (X, y)

    if with_df:
        dfs = {
            name: pd.read_pickle(os.path.join(out_dir, f"{name}_df.pkl"))
            for name in ["train", "val", "test"]
        }
        return (splits["train"][0], splits["train"][1],
                splits["val"][0],   splits["val"][1],
                splits["test"][0],  splits["test"][1],
                dfs["train"], dfs["val"], dfs["test"])

    return (splits["train"][0], splits["train"][1],
            splits["val"][0],   splits["val"][1],
            splits["test"][0],  splits["test"][1])


if __name__ == "__main__":
    main()
