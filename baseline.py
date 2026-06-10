import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from utils import DEVICE, SUPERCLASSES, load_preprocessed, load_xgb_features, \
    PTBXLDataset, model_summary, compute_metrics, print_metrics, \
    tune_thresholds_from_probs, train_one_epoch, evaluate, \
    make_cosine, make_warmup_cosine, save_test_artifacts

NUM_CLASSES = len(SUPERCLASSES)

#################################################
# Baseline 1: XGBoost using hand crafted features
#################################################
XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "early_stopping_rounds": 30,
    "n_jobs": -1, "random_state": 42,
}

def run_xgboost():
    checkpoint_dir = "../checkpoint_xgb"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("Baseline 1 — XGBoost")
    print("=" * 60)
    print("\nLoading hand-crafted features...")
    (X_tr, y_tr, X_va, y_va, X_te, y_te, feature_names) = load_xgb_features()
    print(f"  Train {X_tr.shape}  Val {X_va.shape}  Test {X_te.shape}")
    print(f"  Number of features: {len(feature_names)}")

    for X in (X_tr, X_va, X_te):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    print("\nTraining one XGBoost model per class (One-vs-Rest)...")
    models = {}
    val_probs = np.zeros_like(y_va)
    test_probs = np.zeros_like(y_te)
    for c, cls_name in enumerate(SUPERCLASSES):
        pos = int(y_tr[:, c].sum())
        neg = len(y_tr) - pos
        params = dict(XGB_PARAMS, scale_pos_weight=neg / max(1, pos))
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr[:, c], eval_set=[(X_va, y_va[:, c])], verbose=False)
        val_probs[:, c] = model.predict_proba(X_va)[:, 1]
        test_probs[:, c] = model.predict_proba(X_te)[:, 1]
        model.save_model(os.path.join(checkpoint_dir, f"xgb_{cls_name}.json"))
        models[cls_name] = model
        print(f"  [{c + 1}/{NUM_CLASSES}] {cls_name:5s}  best_iter={model.best_iteration}")

    print("\nTuning per-class thresholds on validation set...")
    thresholds = tune_thresholds_from_probs(val_probs, y_va, NUM_CLASSES)
    for cls, t in zip(SUPERCLASSES, thresholds):
        print(f"    {cls:5s}: {t:.3f}")
    np.save(os.path.join(checkpoint_dir, "thresholds.npy"), thresholds)

    test_metrics = compute_metrics(test_probs, y_te, thresholds=thresholds)
    print_metrics(test_metrics, title="Final Test Results — XGBoost")
    save_test_artifacts(checkpoint_dir, test_metrics, test_probs, y_te)

    # feature importance averaged over the 5 OvR classifiers
    importance = np.mean([m.feature_importances_ for m in models.values()], axis=0)
    pd.DataFrame({"feature": feature_names, "importance": importance}) \
        .sort_values("importance", ascending=False) \
        .to_csv(os.path.join(checkpoint_dir, "feature_importance.csv"), index=False)
    print(f"\nArtifacts saved to {checkpoint_dir}/")
    return test_metrics

####################
# Baseline 2: 1D CNN
####################
class CNN(nn.Module):
    """Four Conv-BN-ReLU-MaxPool blocks -> global avg pool -> MLP head."""

    def __init__(self, num_leads=12, num_classes=5, base_channels=32, dropout=0.3):
        super().__init__()

        def block(in_ch, out_ch, k=7, pool=2):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool),
            )

        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)
        self.features = nn.Sequential(
            block(num_leads, c1, k=7, pool=2),   # (B,  32, 500)
            block(c1, c2, k=5, pool=2),   # (B,  64, 250)
            block(c2, c3, k=5, pool=2),   # (B, 128, 125)
            block(c3, c4, k=3, pool=2),   # (B, 256,  62)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, c4 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c4 // 2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)

######################
# Baseline 3: ResNet1D
######################
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResNet1D(nn.Module):
    def __init__(self, num_leads=12, num_classes=5, stem_channels=64,
                 stage_channels=(64, 128, 256, 512),
                 blocks_per_stage=(2, 2, 2, 2), dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, stem_channels, kernel_size=15, stride=2,
                      padding=7, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )                                              # 12x1000 -> 64x250

        in_ch = stem_channels
        stages = []
        for stage_idx, (out_ch, n_blocks) in enumerate(zip(stage_channels,
                                                            blocks_per_stage)):
            stride = 1 if stage_idx == 0 else 2
            stages.append(BasicBlock1D(in_ch, out_ch, stride=stride))
            for _ in range(n_blocks - 1):
                stages.append(BasicBlock1D(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

#########################
# Baseline 4: Transformer
#########################
class Transformer(nn.Module):
    def __init__(self, num_leads=12, seq_len=1000, patch_size=10, num_classes=5,
                 d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        self.patch_embed = nn.Linear(num_leads * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        x = x.unfold(-1, self.patch_size, self.patch_size)   # (B,12,n_patch,patch)
        x = x.permute(0, 2, 1, 3).contiguous()               # (B,n_patch,12,patch)
        x = x.reshape(B, self.num_patches, -1)               # (B,n_patch,12*patch)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.embed_dropout(x)

        x = self.transformer(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])


#######################################
# Training 
#######################################
def train_torch_baseline(model, cfg, loaders, title):
    """
    Train a torch model with early stopping on validation Macro F1, then
    evaluate the best checkpoint on the test set.
    """
    train_loader, val_loader, test_loader = loaders
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_mdl = os.path.join(cfg["checkpoint_dir"], "best.pt")
    history_path = os.path.join(cfg["checkpoint_dir"], "history.json")

    model = model.to(DEVICE)
    model_summary(model, name=title)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    if cfg.get("warmup_epochs"):
        scheduler = make_warmup_cosine(optimizer, cfg["warmup_epochs"], cfg["epochs"])
    else:
        scheduler = make_cosine(optimizer, cfg["epochs"])

    history = {k: [] for k in ("train_loss", "train_f1", "val_loss", "val_f1",
                               "val_precision", "val_recall", "val_auroc")}
    best_f1, patience = 0.0, 0

    print(f"\nStarting training — {title}")
    for epoch in range(cfg["epochs"]):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, grad_clip=cfg["grad_clip"])
        val_metrics, _, _ = evaluate(model, val_loader, criterion)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_auroc"].append(val_metrics["auroc"])

        print(f"\nEpoch {epoch + 1:02d}/{cfg['epochs']}  (lr={lr_now:.2e})")
        print(f"  Train  Loss={train_loss:.4f}  F1={train_f1:.4f}")
        print(f"  Val    Loss={val_metrics['loss']:.4f}  F1={val_metrics['f1']:.4f}  "
              f"Prec={val_metrics['precision']:.4f}  Recall={val_metrics['recall']:.4f}  "
              f"AUROC={val_metrics['auroc']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1, patience = val_metrics["f1"], 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_f1": best_f1}, best_mdl)
            print(f"   checkpoint: best model saved (val F1: {best_f1:.4f})")
        else:
            patience += 1
            print(f"  Patience {patience}/{cfg['patience']}")
            if patience >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print("\nEvaluating best model on test set...")
    ckpt = torch.load(best_mdl, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_preds, test_targs = evaluate(model, test_loader, criterion)

    print_metrics(test_metrics, title=f"FINAL TEST RESULTS — {title}")
    save_test_artifacts(cfg["checkpoint_dir"], test_metrics, test_preds, test_targs)
    print(f"\nArtifacts saved to {cfg['checkpoint_dir']}/")
    return test_metrics


def make_loaders(augment_train, lead_dropout=True):
    """Build train/val/test DataLoaders from the preprocessed arrays."""
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_preprocessed()
    print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
    pin = (DEVICE == "cuda")
    train = DataLoader(
        PTBXLDataset(X_tr, y_tr, augment=augment_train, lead_dropout=lead_dropout),
        batch_size=64, shuffle=True, num_workers=2, pin_memory=pin)
    val = DataLoader(PTBXLDataset(X_va, y_va, augment=False),
                     batch_size=64, shuffle=False, num_workers=2, pin_memory=pin)
    test = DataLoader(PTBXLDataset(X_te, y_te, augment=False),
                      batch_size=64, shuffle=False, num_workers=2, pin_memory=pin)
    return train, val, test