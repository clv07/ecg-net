import os
import json
import math
import ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch

import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

DATA_DIR = "./ptb-xl/"
OUT_DIR = "./preprocessed/"
FS = 100
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# Bandpass filter cutoffs
BANDPASS_LOW = 0.5   # removes respiratory drift
BANDPASS_HIGH = 40    # removes muscle / high-frequency noise
BANDPASS_ORDER = 4

###############
# Dataset
###############
# Step 1 — Load metadata and aggregate diagnostic superclass
def load_metadata(data_dir):
    print("[1/5] Loading metadata...")
    df = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"))
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    agg = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    agg = agg[agg.diagnostic == 1]

    def aggregate(y_dic):
        return list({agg.loc[k].diagnostic_class for k in y_dic if k in agg.index})

    df["superclass"] = df.scp_codes.apply(aggregate)
    df["has_valid"]  = df.superclass.apply(lambda y: any(c in SUPERCLASSES for c in y))
    return df

# Step 2 — Load raw WFDB signals
def load_signals(df, data_dir, sampling_rate):
    print("[2/5] Loading raw WFDB signals...")
    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr
    signals = []
    for f in tqdm(filenames, desc="WFDB"):
        sig, _ = wfdb.rdsamp(os.path.join(data_dir, f))
        signals.append(sig.T)   # (12, T)
    return np.array(signals, dtype=np.float32)

# Step 3 — Bandpass filter
def bandpass_filter(signals, low, high, fs, order):
    print(f"[3/5] Bandpass filtering {low}-{high} Hz...")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    out = np.empty_like(signals)
    for i in tqdm(range(len(signals)), desc="Filter"):
        out[i] = sosfiltfilt(sos, signals[i], axis=-1).astype(np.float32)
    return out

# Step 4 — Train/val/test split using PTB-XL recommended folds - 1-8 train, 9 val, 10 test
def split_data(X, y, df):
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


# Step 5 — Per-lead z-score normalization using train statistics
def normalize(splits):
    """Compute mean/std from training data only, then apply to all splits."""
    print("[5/5] Per-lead z-score normalization (using train stats)...")
    X_train = splits["train"][0]
    mean = X_train.mean(axis=(0, 2), keepdims=True)   # (1, 12, 1)
    std  = X_train.std(axis=(0, 2), keepdims=True)
    print(f"   per-lead mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"   per-lead std  range: [{std.min():.4f}, {std.max():.4f}]")

    out = {}
    for name, (X, y, df_split) in splits.items():
        out[name] = (((X - mean) / (std + 1e-8)).astype(np.float32), y, df_split)

    np.save(os.path.join(OUT_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(OUT_DIR, "norm_std.npy"),  std)
    return out

# Preprocess data
def preprocess_signals():
    """Full signal-preprocessing pipeline -> cached .npy arrays."""
    print("=" * 60)
    print("PTB-XL Signal Preprocessing")
    print("=" * 60)

    if all(os.path.exists(os.path.join(OUT_DIR, f"{s}_X.npy"))
           for s in ["train", "val", "test"]):
        print("Preprocessed files already exist in", OUT_DIR)
        print("Delete them to regenerate. Skipping.")
        return

    df = load_metadata(DATA_DIR)
    print(f"   total records : {len(df):,}")
    print(f"   valid records : {df.has_valid.sum():,}")

    signals = load_signals(df, DATA_DIR, FS)
    print(f"   raw shape     : {signals.shape}")

    mask = df.has_valid.values
    signals = signals[mask]
    df = df[mask].reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
    y = mlb.fit_transform(df.superclass).astype(np.float32)

    signals = bandpass_filter(signals, BANDPASS_LOW, BANDPASS_HIGH,
                              FS, BANDPASS_ORDER)
    splits = split_data(signals, y, df)
    splits = normalize(splits)

    print("\nSaving preprocessed arrays to", OUT_DIR)
    for name, (X, y_split, df_split) in splits.items():
        np.save(os.path.join(OUT_DIR, f"{name}_X.npy"), X)
        np.save(os.path.join(OUT_DIR, f"{name}_y.npy"), y_split)
        df_split.to_pickle(os.path.join(OUT_DIR, f"{name}_df.pkl"))
        print(f"   {name}: X.npy ({X.nbytes / 1e6:.1f} MB), y.npy, df.pkl")

    np.save(os.path.join(OUT_DIR, "classes.npy"), np.array(SUPERCLASSES))
    print("\nDone.")


def load_preprocessed(out_dir=OUT_DIR, with_df=False):
    """
    Load preprocessed splits. Use this in every deep-model training script.

    Returns:
    - with_df=False : X_train, y_train, X_val, y_val, X_test, y_test
    - with_df=True : the above + (df_train, df_val, df_test)
    """
    splits = {}
    for name in ["train", "val", "test"]:
        X = np.load(os.path.join(out_dir, f"{name}_X.npy"))
        y = np.load(os.path.join(out_dir, f"{name}_y.npy"))
        splits[name] = (X, y)

    if with_df:
        dfs = {name: pd.read_pickle(os.path.join(out_dir, f"{name}_df.pkl"))
               for name in ["train", "val", "test"]}
        return (splits["train"][0], splits["train"][1],
                splits["val"][0],   splits["val"][1],
                splits["test"][0],  splits["test"][1],
                dfs["train"], dfs["val"], dfs["test"])

    return (splits["train"][0], splits["train"][1],
            splits["val"][0],   splits["val"][1],
            splits["test"][0],  splits["test"][1])


# hand crafted feature on bandpass-filtered unnormalized signal for the XGBoost baseline
def _import_neurokit():
    import neurokit2 as nk
    return nk

def _load_unnormalized_signals():
    """Load bandpass-filtered unnormalized signal."""
    df = load_metadata(DATA_DIR)
    signals = load_signals(df, DATA_DIR, FS)

    mask = df.has_valid.values
    signals = signals[mask]
    df = df[mask].reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
    y = mlb.fit_transform(df.superclass).astype(np.float32)

    signals = bandpass_filter(signals, BANDPASS_LOW, BANDPASS_HIGH,
                              FS, BANDPASS_ORDER)

    splits = split_data(signals, y, df)
    return {name: (X, ys) for name, (X, ys, _) in splits.items()}

# Single-lead statistical / spectral features
def stat_features(lead_signal):
    return {
        "mean": float(np.mean(lead_signal)),
        "std":  float(np.std(lead_signal)),
        "skew": float(stats.skew(lead_signal)),
        "kurt": float(stats.kurtosis(lead_signal)),
        "rms":  float(np.sqrt(np.mean(lead_signal ** 2))),
        "ptp":  float(np.ptp(lead_signal)),   # peak-to-peak amplitude
    }

def spectral_features(lead_signal, fs=FS):
    """Band-power ratios in 4 physiologically motivated bands."""
    f, psd = welch(lead_signal, fs=fs, nperseg=min(256, len(lead_signal)))
    bands = {"vlf": (0.0, 0.5), "lf": (0.5, 5), "mf": (5, 15), "hf": (15, 40)}
    out = {}
    total = np.trapz(psd, f) + 1e-8
    for name, (lo, hi) in bands.items():
        idx = (f >= lo) & (f < hi)
        out[f"bp_{name}"] = float(np.trapz(psd[idx], f[idx]) / total)
    return out

# Rhythm / interval / amplitude / cross-lead features (use Lead II R-peaks)
def hrv_features(lead_ii, nk, fs=FS):
    try:
        _, info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 3:
            raise ValueError
        rr = np.diff(rpeaks) / fs * 1000   # ms
        return {
            "hr_mean":  float(60000 / rr.mean()),
            "rr_mean":  float(rr.mean()),
            "rr_sdnn":  float(rr.std()),
            "rr_rmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0,
            "n_beats":  int(len(rpeaks)),
        }
    except Exception:
        return {"hr_mean": 0, "rr_mean": 0, "rr_sdnn": 0, "rr_rmssd": 0, "n_beats": 0}


def interval_features(lead_ii, nk, fs=FS):
    try:
        _, rpeaks_info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        rpeaks = rpeaks_info["ECG_R_Peaks"]
        if len(rpeaks) < 2:
            raise ValueError
        _, waves = nk.ecg_delineate(lead_ii, rpeaks, sampling_rate=fs, method="dwt")

        def median_interval(start_key, end_key):
            starts = np.array(waves.get(start_key, []), dtype=float)
            ends   = np.array(waves.get(end_key,   []), dtype=float)
            n = min(len(starts), len(ends))
            if n == 0:
                return 0.0
            d = (ends[:n] - starts[:n]) / fs * 1000
            d = d[(d > 0) & (d < 1000)]
            return float(np.median(d)) if len(d) else 0.0

        return {
            "pr_interval":  median_interval("ECG_P_Onsets", "ECG_R_Onsets"),
            "qrs_duration": median_interval("ECG_R_Onsets", "ECG_R_Offsets"),
            "qt_interval":  median_interval("ECG_R_Onsets", "ECG_T_Offsets"),
        }
    except Exception:
        return {"pr_interval": 0, "qrs_duration": 0, "qt_interval": 0}


def amplitude_features(record, lead_idx, nk, fs=FS):
    lead_ii = record[1]
    lead    = record[lead_idx]
    try:
        _, info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) == 0:
            raise ValueError

        r_amps = lead[rpeaks]
        st_idx = rpeaks + int(0.06 * fs)
        st_idx = st_idx[st_idx < len(lead)]
        st_levels = lead[st_idx] if len(st_idx) else np.array([0.0])
        t_idx = rpeaks + int(0.2 * fs)
        t_idx = t_idx[t_idx < len(lead)]
        t_amps = lead[t_idx] if len(t_idx) else np.array([0.0])

        return {
            "r_amp":     float(np.median(r_amps)),
            "r_amp_max": float(np.max(np.abs(r_amps))),
            "st_lvl":    float(np.median(st_levels)),
            "t_amp":     float(np.median(t_amps)),
        }
    except Exception:
        return {"r_amp": 0, "r_amp_max": 0, "st_lvl": 0, "t_amp": 0}

def cross_lead_features(record, nk, fs=FS):
    """Clinically motivated features combining multiple leads."""
    try:
        _, info = nk.ecg_peaks(record[1], sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) == 0:
            raise ValueError

        # V1=6, V3=8, V5=10, V6=11, aVL=4
        s_v1 = float(np.median(np.abs(np.minimum(record[6][rpeaks], 0))))
        r_v5 = float(np.median(np.maximum(record[10][rpeaks], 0)))
        r_v6 = float(np.median(np.maximum(record[11][rpeaks], 0)))
        sokolow = s_v1 + max(r_v5, r_v6)            # LVH if > 3.5 mV

        r_avl = float(np.median(np.maximum(record[4][rpeaks], 0)))
        s_v3  = float(np.median(np.abs(np.minimum(record[8][rpeaks], 0))))
        cornell = r_avl + s_v3

        qrs_axis = np.arctan2(np.mean(record[5]), np.mean(record[0])) * 180 / np.pi

        return {"sokolow_lyon": sokolow, "cornell": cornell, "qrs_axis": qrs_axis}
    except Exception:
        return {"sokolow_lyon": 0, "cornell": 0, "qrs_axis": 0}

def extract_features_one(record, nk):
    """record: (12, T) -> flat dict of features."""
    feats = {}
    feats.update(hrv_features(record[1], nk))
    feats.update(interval_features(record[1], nk))
    for i, name in enumerate(LEAD_NAMES):
        for k, v in stat_features(record[i]).items():
            feats[f"{name}_{k}"] = v
        for k, v in spectral_features(record[i]).items():
            feats[f"{name}_{k}"] = v
        for k, v in amplitude_features(record, i, nk).items():
            feats[f"{name}_{k}"] = v
    feats.update(cross_lead_features(record, nk))
    return feats

def extract_features_batch(X, nk, name=""):
    print(f"Extracting features for '{name}' ({len(X)} records)...")
    rows = [extract_features_one(X[i], nk) for i in tqdm(range(len(X)), desc=name)]
    return pd.DataFrame(rows)


def extract_features():
    """Extract hand-crafted features for the XGBoost baseline to .parquet."""
    print("=" * 60)
    print("Hand-crafted Feature Extraction (XGBoost, unnormalized signals)")
    print("=" * 60)

    out_files = [os.path.join(OUT_DIR, f"{s}_features_xgb.parquet")
                 for s in ["train", "val", "test"]]
    if all(os.path.exists(f) for f in out_files):
        print("Feature files already exist. Delete to regenerate. Skipping.")
        return

    nk = _import_neurokit()
    splits = _load_unnormalized_signals()

    feats = {name: extract_features_batch(splits[name][0], nk, name)
             for name in ["train", "val", "test"]}
    print(f"\nFeature count: {feats['train'].shape[1]}")

    for name in ["train", "val", "test"]:
        feats[name].to_parquet(os.path.join(OUT_DIR, f"{name}_features_xgb.parquet"))
        np.save(os.path.join(OUT_DIR, f"{name}_y_xgb.npy"), splits[name][1])

    print("\nSaved feature parquets and labels to", OUT_DIR)


def load_xgb_features(out_dir=OUT_DIR):
    """Load hand-crafted features + labels for the XGBoost baseline."""
    feats = {name: pd.read_parquet(os.path.join(out_dir, f"{name}_features_xgb.parquet"))
             for name in ["train", "val", "test"]}
    ys = {name: np.load(os.path.join(out_dir, f"{name}_y_xgb.npy"))
          for name in ["train", "val", "test"]}
    return (feats["train"].values, ys["train"],
            feats["val"].values,   ys["val"],
            feats["test"].values,  ys["test"],
            list(feats["train"].columns))

class PTBXLDataset(Dataset):
    """
    Wraps the preprocessed (N, 12, 1000) signal arrays and (N, 5) label arrays.

    When augment=True applies label-preserving augmentation for training data only:
      - small additive Gaussian noise (p=0.5)
      - random circular time shift within +/-50 samples (p=0.5)
      - lead dropout: zero out one random lead (p=0.2)
    """

    def __init__(self, X, y, augment=False, lead_dropout=True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment = augment
        self.lead_dropout = lead_dropout

    def __len__(self):
        return len(self.X)

    def _augment(self, x):
        x = x.copy()
        if np.random.rand() < 0.5:
            x += np.random.randn(*x.shape).astype(np.float32) * 0.02
        if np.random.rand() < 0.5:
            shift = np.random.randint(-50, 50)
            x = np.roll(x, shift, axis=-1)
        if self.lead_dropout and np.random.rand() < 0.2:
            x[np.random.randint(0, 12)] = 0
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


###############
# Model Summary
###############
def model_summary(model, name="Model", input_shape=(1, 12, 1000), device=None):
    """
    Report the model architecture, number of trainable parameter, input and output shape.
    """
    device = device or next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"Model Architecture: {name}")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"  Trainable parameters: {n_params:,}")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        out = model(dummy)
    print(f"Input shape : {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    if was_training:
        model.train()
    print("=" * 60)
    return n_params

####################################
# Metrics: Macro F1 and Macro AUROC
####################################
def compute_metrics(probs, targets, thresholds=None):
    """
    Multi-label classification metrics.

    Parameters:
    - probs : (N, C) array of predicted probabilities after sigmoid.
    - targets : (N, C) array of 0/1 ground-truth labels.
    - thresholds : optional (C,) per-class decision thresholds, default 0.5.

    Returns a dict including Macro F1 score, precision, recall and AUROC, 
    and per-class AUROC.
    """
    if thresholds is None:
        binary = (probs > 0.5).astype(int)
    else:
        binary = (probs > np.asarray(thresholds)[None, :]).astype(int)

    per_class_auroc = {}
    for i, name in enumerate(SUPERCLASSES):
        per_class_auroc[name] = roc_auc_score(targets[:, i], probs[:, i])

    metrics = {
        "f1": f1_score(targets, binary, average="macro", zero_division=0),
        "precision": precision_score(targets, binary, average="macro", zero_division=0),
        "recall": recall_score(targets, binary, average="macro", zero_division=0),
        "auroc": roc_auc_score(targets, probs, average="macro"),
        "per_class_auroc": per_class_auroc,
    }

    return metrics


def print_metrics(metrics, title=None):
    if title:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
    if "loss" in metrics:
        print(f"  Loss      : {metrics['loss']:.4f}")
    print(f"  Macro F1  : {metrics['f1']:.4f}")
    print(f"  Macro AUROC: {metrics['auroc']:.4f}")
    print(f"  Macro Prec: {metrics['precision']:.4f}")
    print(f"  Macro Rec : {metrics['recall']:.4f}")
    if "per_class_auroc" in metrics:
        print("  Per-class AUROC:")
        for cls, auc in metrics["per_class_auroc"].items():
            print(f"    {cls:5s}: {auc:.4f}")

###############################################
# Training and evaluation loops (torch models)
###############################################
def train_one_epoch(model, loader, criterion, optimizer, grad_clip=1.0, device=DEVICE):
    model.train()
    running = 0.0
    preds_all, targs_all = [], []
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running += loss.item()
        preds_all.append(torch.sigmoid(out).detach().cpu().numpy())
        targs_all.append(y.detach().cpu().numpy())

    preds = np.vstack(preds_all)
    targs = np.vstack(targs_all)
    f1 = f1_score(targs, (preds > 0.5).astype(int), average="macro", zero_division=0)
    return running / len(loader), f1


def evaluate(model, loader, criterion, thresholds=None, device=DEVICE):
    """
    Run the model and return (metrics, probs, targets).
    """
    model.eval()
    running = 0.0
    preds_all, targs_all = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval ", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            running += criterion(out, y).item()
            preds_all.append(torch.sigmoid(out).cpu().numpy())
            targs_all.append(y.cpu().numpy())

    preds = np.vstack(preds_all)
    targs = np.vstack(targs_all)
    metrics = compute_metrics(preds, targs, thresholds=thresholds)
    metrics["loss"] = running / len(loader)
    return metrics, preds, targs


##################
# Threshold tuning 
##################
def tune_thresholds_from_probs(probs, targets, num_classes=len(SUPERCLASSES)):
    """Per-class F1-maximizing thresholds from probability/target arrays for XGBoost baseline.
    """
    thresholds = np.full(num_classes, 0.5)
    candidates = np.linspace(0.05, 0.95, 91)
    for c in range(num_classes):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            f1 = f1_score(targets[:, c], (probs[:, c] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def tune_thresholds(model, loader, num_classes=len(SUPERCLASSES), device=DEVICE):
    """Per-class F1-maximizing thresholds for a torch model on a DataLoader."""
    model.eval()
    preds_all, targs_all = [], []
    with torch.no_grad():
        for x, y in loader:
            preds_all.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            targs_all.append(y.numpy())
    return tune_thresholds_from_probs(np.vstack(preds_all), np.vstack(targs_all),
                                      num_classes)


###############
# LR Schedules
###############
def make_cosine(optimizer, total_epochs):
    """Plain cosine annealing over total_epochs."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

def make_warmup_cosine(optimizer, warmup_epochs, total_epochs):
    """Linear warmup for warmup_epochs, then cosine decay to zero."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

################
# Save artifacts
################
def save_test_artifacts(checkpoint_dir, metrics, preds, targs):
    """save test predictions and metrics."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    np.save(os.path.join(checkpoint_dir, "test_preds.npy"), preds)
    np.save(os.path.join(checkpoint_dir, "test_targs.npy"), targs)
    serializable = {k: v for k, v in metrics.items() if k != "per_class_auroc"}
    serializable["per_class_auroc"] = metrics.get("per_class_auroc", {})
    with open(os.path.join(checkpoint_dir, "test_metrics.json"), "w") as f:
        json.dump(serializable, f, indent=2)