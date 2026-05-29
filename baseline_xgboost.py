"""
Baseline 1: XGBoost with Hand-crafted Features
================================================
The classical ML baseline. Trains one XGBoost model per class
(One-vs-Rest) on ~130 hand-crafted features extracted from the
bandpass-filtered raw ECG signals.

Purpose:
    Establishes the "non-deep-learning" reference point. Lets us
    measure how much benefit deep learning provides on PTB-XL.

Output format matches the other baselines so the final comparison
script can load all five models uniformly.

Prerequisites:
    1. python preprocess.py            (for label info)
    2. python feature_extract_xgb.py   (for hand-crafted features)

Run:
    python baseline_xgboost.py
"""

import os
import json
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from preprocess import SUPERCLASSES
from feature_extract_xgb import load_xgb_features

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_DIR = "./checkpoint_xgb"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "tree_method":      "hist",
    "early_stopping_rounds": 30,
    "n_jobs":           -1,
    "random_state":     42,
}


# ============================================================
# THRESHOLD TUNING
# ============================================================
def tune_thresholds(probs, targets, num_classes):
    """Per-class threshold tuning on validation predictions."""
    thresholds = np.zeros(num_classes)
    candidates = np.linspace(0.05, 0.95, 91)
    for c in range(num_classes):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            binary = (probs[:, c] > t).astype(int)
            f1 = f1_score(targets[:, c], binary, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def compute_metrics(probs, targets, thresholds=None):
    if thresholds is None:
        binary = (probs > 0.5).astype(int)
    else:
        binary = (probs > thresholds[None, :]).astype(int)

    metrics = {
        "f1":        f1_score(targets, binary, average="macro", zero_division=0),
        "precision": precision_score(targets, binary, average="macro", zero_division=0),
        "recall":    recall_score(targets, binary, average="macro", zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(targets, probs, average="macro")
    except ValueError:
        metrics["auroc"] = 0.0

    per_class_auroc = {}
    for i, name in enumerate(SUPERCLASSES):
        try:
            per_class_auroc[name] = roc_auc_score(targets[:, i], probs[:, i])
        except ValueError:
            per_class_auroc[name] = 0.0
    metrics["per_class_auroc"] = per_class_auroc
    return metrics


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("XGBoost Baseline with Hand-crafted Features")
    print("=" * 60)

    print("\nLoading hand-crafted features...")
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     feature_names) = load_xgb_features()

    print(f"  Train: X {X_train.shape}, y {y_train.shape}")
    print(f"  Val:   X {X_val.shape},   y {y_val.shape}")
    print(f"  Test:  X {X_test.shape},  y {y_test.shape}")
    print(f"  Number of features: {len(feature_names)}")

    # ----- Sanitize: replace inf/nan from feature extraction failures -----
    print("\nSanitizing features (replacing NaN/Inf with 0)...")
    for X in (X_train, X_val, X_test):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- Standardize features (helps XGBoost, no harm if redundant) -----
    print("Standardizing features using training statistics...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # ----- Train one XGBoost per class (OvR) -----
    print("\nTraining one XGBoost model per class (One-vs-Rest)...")
    models = {}
    val_probs = np.zeros_like(y_val)
    test_probs = np.zeros_like(y_test)

    for c, cls_name in enumerate(SUPERCLASSES):
        print(f"\n  [{c + 1}/{len(SUPERCLASSES)}] Class: {cls_name}")
        pos_count = int(y_train[:, c].sum())
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(1, pos_count)
        print(f"     Positives: {pos_count}, Negatives: {neg_count}, "
              f"scale_pos_weight={scale_pos_weight:.2f}")

        # Per-class scale_pos_weight handles imbalance; less destructive
        # than naive pos_weight in BCE for deep models
        params = dict(XGB_PARAMS, scale_pos_weight=scale_pos_weight)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_scaled, y_train[:, c],
            eval_set=[(X_val_scaled, y_val[:, c])],
            verbose=False,
        )
        best_iter = model.best_iteration
        print(f"     Best iteration: {best_iter} / {XGB_PARAMS['n_estimators']}")

        val_probs[:, c]  = model.predict_proba(X_val_scaled)[:, 1]
        test_probs[:, c] = model.predict_proba(X_test_scaled)[:, 1]

        # Save individual model
        model.save_model(os.path.join(CHECKPOINT_DIR, f"xgb_{cls_name}.json"))
        models[cls_name] = model

    # ----- Per-class threshold tuning on validation set -----
    print("\nTuning per-class thresholds on validation set...")
    thresholds = tune_thresholds(val_probs, y_val, len(SUPERCLASSES))
    print("  Tuned thresholds:")
    for cls, t in zip(SUPERCLASSES, thresholds):
        print(f"    {cls:5s}: {t:.3f}")
    np.save(os.path.join(CHECKPOINT_DIR, "thresholds.npy"), thresholds)

    # ----- Evaluation -----
    print("\nValidation metrics (tuned thresholds):")
    val_metrics = compute_metrics(val_probs, y_val, thresholds=thresholds)
    print(f"  Macro F1   : {val_metrics['f1']:.4f}")
    print(f"  Macro AUROC: {val_metrics['auroc']:.4f}")

    print("\nTest with default threshold (0.5):")
    test_default = compute_metrics(test_probs, y_test, thresholds=None)
    print(f"  Macro F1={test_default['f1']:.4f}  "
          f"Prec={test_default['precision']:.4f}  "
          f"Recall={test_default['recall']:.4f}")

    print("\nTest with tuned per-class thresholds:")
    test_metrics = compute_metrics(test_probs, y_test, thresholds=thresholds)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS — XGBoost (hand-crafted features)")
    print("=" * 60)
    print(f"  Macro F1  : {test_metrics['f1']:.4f}")
    print(f"  Macro Prec: {test_metrics['precision']:.4f}")
    print(f"  Macro Rec : {test_metrics['recall']:.4f}")
    print(f"  Macro AUC : {test_metrics['auroc']:.4f}")
    print("  Per-class AUROC:")
    for cls, auc in test_metrics["per_class_auroc"].items():
        print(f"    {cls:5s}: {auc:.4f}")

    # ----- Save artifacts for downstream comparison -----
    np.save(os.path.join(CHECKPOINT_DIR, "test_preds.npy"), test_probs)
    np.save(os.path.join(CHECKPOINT_DIR, "test_targs.npy"), y_test)
    with open(os.path.join(CHECKPOINT_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if k != "per_class_auroc"} | {"per_class_auroc": test_metrics["per_class_auroc"]},
                  f, indent=2)

    # ----- Feature importance (aggregated across the 5 OvR classifiers) -----
    print("\nTop 20 features by importance (averaged across 5 classifiers):")
    importance = np.zeros(len(feature_names))
    for model in models.values():
        importance += model.feature_importances_
    importance /= len(models)
    top_idx = np.argsort(importance)[::-1][:20]
    for i in top_idx:
        print(f"    {feature_names[i]:30s}  {importance[i]:.4f}")

    # save feature importance for the report
    import pandas as pd
    pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).to_csv(
        os.path.join(CHECKPOINT_DIR, "feature_importance.csv"),
        index=False,
    )

    print(f"\nArtifacts saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
