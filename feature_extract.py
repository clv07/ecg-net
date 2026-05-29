"""
Hand-crafted Feature Extraction for PTB-XL
============================================
Sits on top of the unified preprocessed signals (from preprocess.py) and
extracts clinical/statistical features for the XGBoost baseline.

Features per record:
  - Heart rate statistics (mean HR, HRV: SDNN, RMSSD)
  - Interval features (PR, QRS width, QT) — extracted from Lead II
  - Amplitude features per lead (R-peak amplitude, ST level, T-wave amplitude)
  - Spectral features per lead (band power in 4 freq bands)
  - Statistical features per lead (mean, std, skew, kurtosis, RMS)
  - Cross-lead features (Sokolow-Lyon index for HYP)

Output: feature matrix (N, ~120) — exact count printed at end.

Run after preprocess.py:
    python feature_extract.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats, signal as sp_signal
from scipy.signal import welch

import neurokit2 as nk   # pip install neurokit2

from preprocess import load_preprocessed, OUT_DIR, LEAD_NAMES

warnings.filterwarnings("ignore")

FS = 100   # sampling rate

# ============================================================
# Single-lead statistical features
# ============================================================
def stat_features(lead_signal):
    """5 simple statistical features per lead."""
    return {
        "mean": float(np.mean(lead_signal)),
        "std":  float(np.std(lead_signal)),
        "skew": float(stats.skew(lead_signal)),
        "kurt": float(stats.kurtosis(lead_signal)),
        "rms":  float(np.sqrt(np.mean(lead_signal ** 2))),
    }


# ============================================================
# Spectral features (band power)
# ============================================================
def spectral_features(lead_signal, fs=FS):
    """Band power in 4 physiologically motivated bands."""
    f, psd = welch(lead_signal, fs=fs, nperseg=min(256, len(lead_signal)))
    bands = {
        "vlf":  (0.0, 0.5),    # baseline drift
        "lf":   (0.5, 5),      # P/T waves
        "mf":   (5, 15),       # QRS
        "hf":   (15, 40),      # noise / muscle
    }
    out = {}
    total = np.trapz(psd, f) + 1e-8
    for name, (lo, hi) in bands.items():
        idx = (f >= lo) & (f < hi)
        out[f"bp_{name}"] = float(np.trapz(psd[idx], f[idx]) / total)
    return out


# ============================================================
# R-peak based features (heart rate, HRV) — uses Lead II
# ============================================================
def hrv_features(lead_ii, fs=FS):
    """Heart rate and rhythm features from Lead II."""
    try:
        _, info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 3:
            raise ValueError("too few R-peaks")
        rr = np.diff(rpeaks) / fs * 1000   # in ms
        return {
            "hr_mean":  float(60000 / rr.mean()),
            "hr_std":   float(60000 / rr.std() if rr.std() > 0 else 0),
            "rr_mean":  float(rr.mean()),
            "rr_sdnn":  float(rr.std()),
            "rr_rmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0,
            "n_beats":  int(len(rpeaks)),
        }
    except Exception:
        return {"hr_mean": 0, "hr_std": 0, "rr_mean": 0,
                "rr_sdnn": 0, "rr_rmssd": 0, "n_beats": 0}


# ============================================================
# Interval features (PR, QRS, QT) — uses Lead II
# ============================================================
def interval_features(lead_ii, fs=FS):
    """
    Extract PR, QRS, QT intervals using neurokit2 wave delineation.
    Returns median values across all detected beats.
    """
    try:
        _, rpeaks_info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        rpeaks = rpeaks_info["ECG_R_Peaks"]
        if len(rpeaks) < 2:
            raise ValueError("too few beats")

        _, waves = nk.ecg_delineate(lead_ii, rpeaks,
                                     sampling_rate=fs, method="dwt")

        def median_interval(start_key, end_key):
            starts = np.array(waves.get(start_key, []), dtype=float)
            ends   = np.array(waves.get(end_key,   []), dtype=float)
            n = min(len(starts), len(ends))
            if n == 0:
                return 0.0
            diffs = (ends[:n] - starts[:n]) / fs * 1000   # ms
            diffs = diffs[(diffs > 0) & (diffs < 1000)]   # sanity
            return float(np.median(diffs)) if len(diffs) else 0.0

        return {
            "pr_interval":  median_interval("ECG_P_Onsets",   "ECG_R_Onsets"),
            "qrs_duration": median_interval("ECG_R_Onsets",   "ECG_R_Offsets"),
            "qt_interval":  median_interval("ECG_R_Onsets",   "ECG_T_Offsets"),
        }
    except Exception:
        return {"pr_interval": 0, "qrs_duration": 0, "qt_interval": 0}


# ============================================================
# Amplitude features per lead (R, ST, T)
# ============================================================
def amplitude_features(record, lead_idx, fs=FS):
    """
    R-peak amplitude (max), ST-segment level (J+60ms), T-wave amplitude.
    R-peaks detected from Lead II then used to look up all 12 leads.
    """
    lead_ii = record[1]
    lead    = record[lead_idx]
    try:
        _, info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) == 0:
            raise ValueError
        # R amplitude on this lead = signal value at R-peak positions
        r_amps = lead[rpeaks]
        # ST level: 60 ms after R-peak
        st_idx = rpeaks + int(0.06 * fs)
        st_idx = st_idx[st_idx < len(lead)]
        st_levels = lead[st_idx] if len(st_idx) else np.array([0.0])
        # T amplitude: 200 ms after R-peak (rough)
        t_idx = rpeaks + int(0.2 * fs)
        t_idx = t_idx[t_idx < len(lead)]
        t_amps = lead[t_idx] if len(t_idx) else np.array([0.0])

        return {
            "r_amp":  float(np.median(r_amps)),
            "st_lvl": float(np.median(st_levels)),
            "t_amp":  float(np.median(t_amps)),
        }
    except Exception:
        return {"r_amp": 0, "st_lvl": 0, "t_amp": 0}


# ============================================================
# Cross-lead features
# ============================================================
def cross_lead_features(record, fs=FS):
    """
    Features that combine multiple leads.
    Sokolow-Lyon: S(V1) + max(R(V5), R(V6))  →  LVH indicator
    """
    try:
        _, info = nk.ecg_peaks(record[1], sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) == 0:
            raise ValueError

        # V1=idx 6, V5=10, V6=11
        s_v1 = float(np.median(np.abs(record[6][rpeaks].min(initial=0))))
        r_v5 = float(np.median(record[10][rpeaks].max(initial=0)))
        r_v6 = float(np.median(record[11][rpeaks].max(initial=0)))

        return {
            "sokolow_lyon": s_v1 + max(r_v5, r_v6),
            "qrs_axis":     np.arctan2(np.mean(record[5]),    # aVF
                                       np.mean(record[0])) * 180 / np.pi,  # I
        }
    except Exception:
        return {"sokolow_lyon": 0, "qrs_axis": 0}


# ============================================================
# Per-record feature extraction
# ============================================================
def extract_features_one(record):
    """
    record: (12, T) numpy array
    Returns: flat dict of features
    """
    feats = {}

    # Lead II (idx 1) — global rhythm/interval features
    feats.update(hrv_features(record[1]))
    feats.update(interval_features(record[1]))

    # Per-lead stats, spectral, amplitude
    for i, name in enumerate(LEAD_NAMES):
        for k, v in stat_features(record[i]).items():
            feats[f"{name}_{k}"] = v
        for k, v in spectral_features(record[i]).items():
            feats[f"{name}_{k}"] = v
        for k, v in amplitude_features(record, i).items():
            feats[f"{name}_{k}"] = v

    # Cross-lead
    feats.update(cross_lead_features(record))

    return feats


# ============================================================
# Batch extraction
# ============================================================
def extract_features_batch(X, name=""):
    """X: (N, 12, T)  →  DataFrame of features."""
    print(f"Extracting features for '{name}' ({len(X)} records)...")
    rows = []
    for i in tqdm(range(len(X)), desc=name):
        rows.append(extract_features_one(X[i]))
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Hand-crafted Feature Extraction")
    print("=" * 60)

    feat_files = [os.path.join(OUT_DIR, f"{s}_features.parquet")
                  for s in ["train", "val", "test"]]
    if all(os.path.exists(f) for f in feat_files):
        print("Feature files already exist. Delete to regenerate. Exiting.")
        return

    print("Loading preprocessed signals...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed()

    feats_train = extract_features_batch(X_train, "train")
    feats_val   = extract_features_batch(X_val,   "val")
    feats_test  = extract_features_batch(X_test,  "test")

    print(f"\nFeature count: {feats_train.shape[1]}")
    print(f"Feature names (sample): {list(feats_train.columns[:5])} ...")

    # save as parquet (compact, fast load)
    feats_train.to_parquet(os.path.join(OUT_DIR, "train_features.parquet"))
    feats_val.to_parquet(  os.path.join(OUT_DIR, "val_features.parquet"))
    feats_test.to_parquet( os.path.join(OUT_DIR, "test_features.parquet"))

    print("\nSaved feature parquets to", OUT_DIR)


# ============================================================
# LOADER
# ============================================================
def load_features(out_dir=OUT_DIR):
    """Load hand-crafted features + labels for XGBoost training."""
    feats_train = pd.read_parquet(os.path.join(out_dir, "train_features.parquet"))
    feats_val   = pd.read_parquet(os.path.join(out_dir, "val_features.parquet"))
    feats_test  = pd.read_parquet(os.path.join(out_dir, "test_features.parquet"))
    y_train = np.load(os.path.join(out_dir, "train_y.npy"))
    y_val   = np.load(os.path.join(out_dir, "val_y.npy"))
    y_test  = np.load(os.path.join(out_dir, "test_y.npy"))
    return (feats_train.values, y_train,
            feats_val.values,   y_val,
            feats_test.values,  y_test,
            list(feats_train.columns))


if __name__ == "__main__":
    main()
