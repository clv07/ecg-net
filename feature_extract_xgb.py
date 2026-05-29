"""
Hand-crafted Feature Extraction for XGBoost Baseline
=====================================================
IMPORTANT: This version operates on bandpass-filtered but UNNORMALIZED
signals (i.e., raw mV amplitudes preserved).

Why: Many ECG diagnostic features are amplitude-based:
  - R-wave amplitude  → Sokolow-Lyon for hypertrophy (HYP)
  - ST segment level  → MI diagnosis
  - QRS voltage       → axis deviation, HYP
Per-lead z-score normalization (as in preprocess.py) destroys these
absolute amplitude relationships, which makes the features less useful.

For deep models, the network can learn to be amplitude-invariant or
not, depending on what helps. For XGBoost with hand-crafted features,
amplitude is part of the feature definition and must be preserved.

Output: feature matrices for train/val/test, saved as .parquet files.

Run after preprocess.py has been run (we reuse its split logic):
    python feature_extract_xgb.py
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.preprocessing import MultiLabelBinarizer

import neurokit2 as nk

from preprocess import (
    DATA_DIR, OUT_DIR, SAMPLING_RATE,
    SUPERCLASSES, LEAD_NAMES,
    BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
)

warnings.filterwarnings("ignore")

FS = SAMPLING_RATE


# ============================================================
# REUSE STEPS 1-3 FROM preprocess.py (but skip normalization)
# ============================================================
def load_unnormalized_signals():
    """
    Load PTB-XL signals with bandpass filter applied but NO normalization.
    Returns X, y, df split by PTB-XL recommended folds.
    """
    # ----- metadata -----
    df = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"))
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    agg = pd.read_csv(os.path.join(DATA_DIR, "scp_statements.csv"), index_col=0)
    agg = agg[agg.diagnostic == 1]

    def aggregate(y_dic):
        return list({agg.loc[k].diagnostic_class
                     for k in y_dic if k in agg.index})

    df["superclass"] = df.scp_codes.apply(aggregate)
    df["has_valid"]  = df.superclass.apply(
        lambda y: any(c in SUPERCLASSES for c in y)
    )

    # ----- signals -----
    print("Loading raw WFDB signals...")
    filenames = df.filename_lr if SAMPLING_RATE == 100 else df.filename_hr
    signals = []
    for f in tqdm(filenames, desc="WFDB"):
        sig, _ = wfdb.rdsamp(os.path.join(DATA_DIR, f))
        signals.append(sig.T)
    signals = np.array(signals, dtype=np.float32)

    # ----- filter to valid records -----
    mask = df.has_valid.values
    signals = signals[mask]
    df = df[mask].reset_index(drop=True)

    # ----- bandpass filter (no normalization) -----
    print(f"Bandpass {BANDPASS_LOW}-{BANDPASS_HIGH} Hz...")
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH],
                 btype="bandpass", fs=FS, output="sos")
    out = np.empty_like(signals)
    for i in tqdm(range(len(signals)), desc="Filter"):
        out[i] = sosfiltfilt(sos, signals[i], axis=-1).astype(np.float32)
    signals = out

    # ----- labels -----
    mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
    y = mlb.fit_transform(df.superclass).astype(np.float32)

    # ----- splits -----
    splits = {}
    for name, fold_check in [("train", lambda f: f <= 8),
                              ("val",   lambda f: f == 9),
                              ("test",  lambda f: f == 10)]:
        m = df.strat_fold.apply(fold_check).values
        splits[name] = (signals[m], y[m])
        print(f"  {name}: {m.sum()} records")

    return splits


# ============================================================
# FEATURE EXTRACTORS
# ============================================================
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
    f, psd = welch(lead_signal, fs=fs, nperseg=min(256, len(lead_signal)))
    bands = {"vlf": (0.0, 0.5), "lf": (0.5, 5),
             "mf":  (5, 15),    "hf": (15, 40)}
    out = {}
    total = np.trapz(psd, f) + 1e-8
    for name, (lo, hi) in bands.items():
        idx = (f >= lo) & (f < hi)
        out[f"bp_{name}"] = float(np.trapz(psd[idx], f[idx]) / total)
    return out


def hrv_features(lead_ii, fs=FS):
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
        return {"hr_mean": 0, "rr_mean": 0,
                "rr_sdnn": 0, "rr_rmssd": 0, "n_beats": 0}


def interval_features(lead_ii, fs=FS):
    try:
        _, rpeaks_info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        rpeaks = rpeaks_info["ECG_R_Peaks"]
        if len(rpeaks) < 2:
            raise ValueError
        _, waves = nk.ecg_delineate(lead_ii, rpeaks,
                                     sampling_rate=fs, method="dwt")

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


def amplitude_features(record, lead_idx, fs=FS):
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


def cross_lead_features(record, fs=FS):
    """Features that combine multiple leads — clinically motivated."""
    try:
        _, info = nk.ecg_peaks(record[1], sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) == 0:
            raise ValueError

        # V1=6, V5=10, V6=11
        s_v1 = float(np.median(np.abs(np.minimum(record[6][rpeaks], 0))))
        r_v5 = float(np.median(np.maximum(record[10][rpeaks], 0)))
        r_v6 = float(np.median(np.maximum(record[11][rpeaks], 0)))

        # Sokolow-Lyon: S(V1) + max(R(V5), R(V6)) — LVH if > 3.5 mV
        sokolow = s_v1 + max(r_v5, r_v6)

        # Cornell: R(aVL) + S(V3)
        r_avl = float(np.median(np.maximum(record[4][rpeaks], 0)))
        s_v3  = float(np.median(np.abs(np.minimum(record[8][rpeaks], 0))))
        cornell = r_avl + s_v3

        # QRS axis from Lead I (0) and aVF (5)
        qrs_axis = np.arctan2(np.mean(record[5]), np.mean(record[0])) * 180 / np.pi

        return {
            "sokolow_lyon": sokolow,
            "cornell":      cornell,
            "qrs_axis":     qrs_axis,
        }
    except Exception:
        return {"sokolow_lyon": 0, "cornell": 0, "qrs_axis": 0}


def extract_features_one(record):
    feats = {}
    feats.update(hrv_features(record[1]))
    feats.update(interval_features(record[1]))

    for i, name in enumerate(LEAD_NAMES):
        for k, v in stat_features(record[i]).items():
            feats[f"{name}_{k}"] = v
        for k, v in spectral_features(record[i]).items():
            feats[f"{name}_{k}"] = v
        for k, v in amplitude_features(record, i).items():
            feats[f"{name}_{k}"] = v

    feats.update(cross_lead_features(record))
    return feats


def extract_features_batch(X, name=""):
    print(f"Extracting features for '{name}' ({len(X)} records)...")
    rows = [extract_features_one(X[i]) for i in tqdm(range(len(X)), desc=name)]
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Hand-crafted Feature Extraction (Unnormalized)")
    print("=" * 60)

    out_files = [os.path.join(OUT_DIR, f"{s}_features_xgb.parquet")
                 for s in ["train", "val", "test"]]
    if all(os.path.exists(f) for f in out_files):
        print("Feature files already exist. Delete to regenerate.")
        return

    splits = load_unnormalized_signals()

    feats_train = extract_features_batch(splits["train"][0], "train")
    feats_val   = extract_features_batch(splits["val"][0],   "val")
    feats_test  = extract_features_batch(splits["test"][0],  "test")

    print(f"\nFeature count: {feats_train.shape[1]}")

    feats_train.to_parquet(os.path.join(OUT_DIR, "train_features_xgb.parquet"))
    feats_val.to_parquet(  os.path.join(OUT_DIR, "val_features_xgb.parquet"))
    feats_test.to_parquet( os.path.join(OUT_DIR, "test_features_xgb.parquet"))

    # Save y too (use same labels as preprocess.py since splits are deterministic)
    np.save(os.path.join(OUT_DIR, "train_y_xgb.npy"), splits["train"][1])
    np.save(os.path.join(OUT_DIR, "val_y_xgb.npy"),   splits["val"][1])
    np.save(os.path.join(OUT_DIR, "test_y_xgb.npy"),  splits["test"][1])

    print("\nSaved feature parquets and labels to", OUT_DIR)


def load_xgb_features(out_dir=OUT_DIR):
    feats_train = pd.read_parquet(os.path.join(out_dir, "train_features_xgb.parquet"))
    feats_val   = pd.read_parquet(os.path.join(out_dir, "val_features_xgb.parquet"))
    feats_test  = pd.read_parquet(os.path.join(out_dir, "test_features_xgb.parquet"))
    y_train = np.load(os.path.join(out_dir, "train_y_xgb.npy"))
    y_val   = np.load(os.path.join(out_dir, "val_y_xgb.npy"))
    y_test  = np.load(os.path.join(out_dir, "test_y_xgb.npy"))
    return (feats_train.values, y_train,
            feats_val.values,   y_val,
            feats_test.values,  y_test,
            list(feats_train.columns))


if __name__ == "__main__":
    main()
