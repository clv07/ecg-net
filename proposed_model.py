"""
Proposed Model: Multi-scale CNN + Lead Attention + Temporal Transformer
========================================================================
Designed to outperform the Vanilla CNN, ResNet1D, and Pure Transformer
baselines on PTB-XL, with particular focus on the two hardest classes:
CD (conduction disturbance) and HYP (hypertrophy).

Architecture rationale:
    - Multi-scale Conv1d frontend captures QRS morphology at multiple
      temporal scales simultaneously (helps CD: bundle branch blocks
      manifest as widened QRS).
    - Lead Attention module explicitly models inter-lead relationships
      (helps HYP: Sokolow-Lyon index depends on V1+V5/V6 amplitudes).
    - Temporal Transformer encoder captures long-range dependencies
      across the 10-second window.

Training enhancements:
    - Per-class pos_weight in BCE loss (addresses class imbalance)
    - Per-class threshold tuning on validation set
    - Warmup + cosine LR schedule
    - Light augmentation (noise, time shift, lead dropout)

Run:
    python proposed_model.py
"""

import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from preprocess import load_preprocessed, SUPERCLASSES

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_DIR = "./checkpoint_proposed"
BEST_MDL       = os.path.join(CHECKPOINT_DIR, "best.pt")
LAST_MDL       = os.path.join(CHECKPOINT_DIR, "last.pt")
HISTORY_PATH   = os.path.join(CHECKPOINT_DIR, "history.json")

# Architecture
D_MODEL          = 192
NHEAD            = 6
NUM_LAYERS       = 3
DIM_FEEDFORWARD  = 384
DROPOUT          = 0.2

# Training
BATCH_SIZE       = 64
EPOCHS           = 50
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 4
EARLY_STOP_PATIENCE = 10
GRAD_CLIP        = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# DATASET (with augmentation)
# ============================================================
class PTBXLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, x):
        x = x.copy()
        # Gaussian noise
        if np.random.rand() < 0.5:
            x += np.random.randn(*x.shape).astype(np.float32) * 0.02
        # Time shift
        if np.random.rand() < 0.5:
            shift = np.random.randint(-50, 50)
            x = np.roll(x, shift, axis=-1)
        # Mild amplitude scaling (small range to preserve ST diagnostic info)
        if np.random.rand() < 0.5:
            x *= np.random.uniform(0.9, 1.1)
        # NOTE: no lead dropout — bad for HYP which relies on V1+V5/V6
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


# ============================================================
# MODULE 1 — Multi-scale CNN frontend
# ============================================================
class MultiScaleConvBlock(nn.Module):
    """
    Three parallel Conv1d branches with different kernel sizes, concatenated.
    - kernel=5:  captures narrow features (R-peak shape, P-wave)
    - kernel=15: captures full QRS width (~150ms at 100Hz)
    - kernel=31: captures broader context (~300ms — P-QRS-T span)

    Useful for CD: bundle branch blocks widen QRS from ~80ms to >120ms;
    multi-scale lets the network learn what "wide QRS" means in context.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # split output channels evenly across three branches
        c = out_channels // 3
        c_last = out_channels - 2 * c   # accommodate remainder

        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=5,
                      stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        self.branch_med = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=15,
                      stride=stride, padding=7, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, c_last, kernel_size=31,
                      stride=stride, padding=15, bias=False),
            nn.BatchNorm1d(c_last),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat([
            self.branch_small(x),
            self.branch_med(x),
            self.branch_large(x),
        ], dim=1)


# ============================================================
# MODULE 2 — Lead Attention
# ============================================================
class LeadAttention(nn.Module):
    """
    Self-attention across the 12 leads (lead-as-token), applied AFTER
    a per-lead convolutional encoding.

    Process:
        Input:  (B, 12, T) — raw signal
        Per-lead Conv:    (B, 12, lead_dim)  — compress time to a fixed embedding per lead
        Lead attention:   (B, 12, lead_dim)  — attend across leads
        Output:           (B, 12, lead_dim)  — lead-aware features

    Helps HYP: the Sokolow-Lyon criterion (S in V1 + R in V5/V6 > 35mm)
    requires explicit cross-lead reasoning. CNN's lead fusion is implicit
    and harder to learn from limited data.
    """
    def __init__(self, num_leads=12, lead_dim=64, nhead=4, dropout=0.1):
        super().__init__()
        # Compress each lead's full time series into a fixed-dim embedding
        self.lead_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),   # (B*12, 32, 1)
            nn.Flatten(),               # (B*12, 32)
            nn.Linear(32, lead_dim),
        )

        self.lead_pos_embed = nn.Parameter(
            torch.randn(1, num_leads, lead_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lead_dim,
            nhead=nhead,
            dim_feedforward=lead_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.lead_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.num_leads = num_leads
        self.lead_dim = lead_dim

    def forward(self, x):
        # x: (B, 12, T)
        B = x.size(0)
        # Encode each lead independently
        x_in = x.unsqueeze(2)                         # (B, 12, 1, T)
        x_in = x_in.view(B * self.num_leads, 1, -1)   # (B*12, 1, T)
        lead_emb = self.lead_encoder(x_in)            # (B*12, lead_dim)
        lead_emb = lead_emb.view(B, self.num_leads, self.lead_dim)

        # Add lead positional embedding (so attention knows which lead is which)
        lead_emb = lead_emb + self.lead_pos_embed

        # Self-attend across leads
        return self.lead_attn(lead_emb)               # (B, 12, lead_dim)


# ============================================================
# MAIN MODEL
# ============================================================
class ProposedECGModel(nn.Module):
    """
    Multi-scale CNN + Lead Attention + Temporal Transformer.
    """
    def __init__(self,
                 num_leads=12,
                 num_classes=5,
                 d_model=D_MODEL,
                 nhead=NHEAD,
                 num_layers=NUM_LAYERS,
                 dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT,
                 lead_dim=64):
        super().__init__()
        self.num_leads = num_leads
        self.lead_dim  = lead_dim

        # --- (1) Lead attention branch — produces a (B, 12*lead_dim) global lead-aware vector ---
        self.lead_attention = LeadAttention(
            num_leads=num_leads, lead_dim=lead_dim, nhead=4, dropout=dropout
        )

        # --- (2) Multi-scale CNN frontend for temporal stream ---
        c1 = 48                                # split as 16/16/16
        c2 = 96
        c3 = d_model
        self.cnn = nn.Sequential(
            MultiScaleConvBlock(num_leads, c1, stride=2),   # 1000 → 500
            nn.MaxPool1d(2),                                # 500 → 250
            MultiScaleConvBlock(c1, c2, stride=2),          # 250 → 125
            nn.MaxPool1d(2),                                # 125 → 62
            MultiScaleConvBlock(c2, c3, stride=1),          # 62 → 62
            nn.Dropout(dropout),
        )
        # output: (B, d_model, ~62)

        # --- (3) Positional embedding for time tokens ---
        self.time_pos_embed = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        # --- (4) Temporal Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        # --- (5) Fuse temporal + lead representations and classify ---
        lead_total_dim = num_leads * lead_dim   # 12 * 64 = 768
        fused_dim = d_model + lead_total_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 12, 1000)
        B = x.size(0)

        # ----- Lead attention stream -----
        lead_features = self.lead_attention(x)            # (B, 12, lead_dim)
        lead_pooled = lead_features.flatten(1)            # (B, 12 * lead_dim)

        # ----- Temporal CNN + Transformer stream -----
        h = self.cnn(x)                                   # (B, d_model, T')
        h = h.permute(0, 2, 1)                            # (B, T', d_model)
        T_prime = h.size(1)
        h = h + self.time_pos_embed[:, :T_prime, :]
        h = self.temporal_transformer(h)
        h = self.temporal_norm(h)
        temporal_pooled = h.mean(dim=1)                   # (B, d_model)

        # ----- Fuse and classify -----
        fused = torch.cat([temporal_pooled, lead_pooled], dim=1)
        return self.classifier(fused)


# ============================================================
# LR SCHEDULER
# ============================================================
def make_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# CLASS WEIGHTING
# ============================================================
def compute_pos_weight(y_train):
    """
    BCEWithLogitsLoss pos_weight = (#negatives / #positives) per class.
    Up-weights minority classes (notably HYP).
    """
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pw = neg / (pos + 1e-8)
    return torch.tensor(pw, dtype=torch.float32)


# ============================================================
# THRESHOLD TUNING
# ============================================================
def tune_thresholds(model, loader, num_classes):
    """
    Find per-class probability threshold that maximizes F1 on validation set.
    Default threshold of 0.5 is rarely optimal under class imbalance.
    """
    model.eval()
    preds_all, targs_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds_all.append(torch.sigmoid(model(x)).cpu().numpy())
            targs_all.append(y.numpy())
    preds = np.vstack(preds_all)
    targs = np.vstack(targs_all)

    thresholds = np.zeros(num_classes)
    candidates = np.linspace(0.05, 0.95, 91)
    for c in range(num_classes):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            binary = (preds[:, c] > t).astype(int)
            f1 = f1_score(targs[:, c], binary, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running = 0.0
    preds_all, targs_all = [], []
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        running += loss.item()
        preds_all.append(torch.sigmoid(out).detach().cpu().numpy())
        targs_all.append(y.detach().cpu().numpy())

    preds = np.vstack(preds_all)
    targs = np.vstack(targs_all)
    f1 = f1_score(targs, (preds > 0.5).astype(int),
                  average="macro", zero_division=0)
    return running / len(loader), f1


def evaluate(model, loader, criterion, thresholds=None):
    """
    If `thresholds` is provided, use per-class thresholds for F1/Prec/Recall.
    AUROC is threshold-independent.
    """
    model.eval()
    running = 0.0
    preds_all, targs_all = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval ", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            running += loss.item()
            preds_all.append(torch.sigmoid(out).cpu().numpy())
            targs_all.append(y.cpu().numpy())

    preds = np.vstack(preds_all)
    targs = np.vstack(targs_all)

    if thresholds is None:
        binary = (preds > 0.5).astype(int)
    else:
        binary = (preds > thresholds[None, :]).astype(int)

    metrics = {
        "loss":      running / len(loader),
        "f1":        f1_score(targs, binary, average="macro", zero_division=0),
        "precision": precision_score(targs, binary, average="macro", zero_division=0),
        "recall":    recall_score(targs, binary, average="macro", zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(targs, preds, average="macro")
    except ValueError:
        metrics["auroc"] = 0.0

    per_class_auroc = {}
    for i, name in enumerate(SUPERCLASSES):
        try:
            per_class_auroc[name] = roc_auc_score(targs[:, i], preds[:, i])
        except ValueError:
            per_class_auroc[name] = 0.0
    metrics["per_class_auroc"] = per_class_auroc

    return metrics, preds, targs


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Device: {DEVICE}")

    print("Loading preprocessed data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed()
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    train_loader = DataLoader(PTBXLDataset(X_train, y_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(PTBXLDataset(X_val,   y_val,   augment=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))
    test_loader  = DataLoader(PTBXLDataset(X_test,  y_test,  augment=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))

    print("\nInitializing proposed model...")
    model = ProposedECGModel(num_classes=len(SUPERCLASSES)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Class-weighted BCE
    pos_weight = compute_pos_weight(y_train).to(DEVICE)
    print(f"  Class pos_weights: {pos_weight.cpu().numpy().round(2)}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    history = {"train_loss": [], "train_f1": [],
               "val_loss":   [], "val_f1":   [],
               "val_precision": [], "val_recall": [], "val_auroc": [],
               "lr": []}

    best_auroc = 0.0   # use AUROC (not F1) for model selection — more stable
    patience = 0

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics, _, _    = evaluate(model, val_loader, criterion)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["lr"].append(lr_now)

        print(f"\nEpoch {epoch + 1:02d}/{EPOCHS}  (lr={lr_now:.2e})")
        print(f"  Train  Loss={train_loss:.4f}  F1={train_f1:.4f}")
        print(f"  Val    Loss={val_metrics['loss']:.4f}  "
              f"F1={val_metrics['f1']:.4f}  "
              f"Prec={val_metrics['precision']:.4f}  "
              f"Recall={val_metrics['recall']:.4f}  "
              f"AUROC={val_metrics['auroc']:.4f}")
        print(f"  Val per-class AUROC: " + "  ".join(
            f"{cls}={v:.3f}" for cls, v in val_metrics["per_class_auroc"].items()
        ))

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
        }, LAST_MDL)

        # Select best by AUROC (threshold-independent, more stable than F1)
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auroc": val_metrics["auroc"],
            }, BEST_MDL)
            print(f"  ✓ Best model saved (val AUROC: {best_auroc:.4f})")
        else:
            patience += 1
            print(f"  Patience {patience}/{EARLY_STOP_PATIENCE}")
            if patience >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    # ----- Load best model -----
    print("\nLoading best model for evaluation...")
    ckpt = torch.load(BEST_MDL, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # ----- Tune per-class thresholds on validation set -----
    print("\nTuning per-class thresholds on validation set...")
    thresholds = tune_thresholds(model, val_loader, len(SUPERCLASSES))
    print("  Tuned thresholds:")
    for cls, t in zip(SUPERCLASSES, thresholds):
        print(f"    {cls:5s}: {t:.3f}")
    np.save(os.path.join(CHECKPOINT_DIR, "thresholds.npy"), thresholds)

    # ----- Test with default and tuned thresholds -----
    print("\nTest with default threshold (0.5):")
    test_metrics_default, _, _ = evaluate(model, test_loader, criterion, thresholds=None)
    print(f"  Macro F1={test_metrics_default['f1']:.4f}  "
          f"Prec={test_metrics_default['precision']:.4f}  "
          f"Recall={test_metrics_default['recall']:.4f}")

    print("\nTest with tuned per-class thresholds:")
    test_metrics, test_preds, test_targs = evaluate(model, test_loader, criterion, thresholds=thresholds)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS — Proposed Model")
    print("=" * 60)
    print(f"  Loss      : {test_metrics['loss']:.4f}")
    print(f"  Macro F1  : {test_metrics['f1']:.4f}")
    print(f"  Macro Prec: {test_metrics['precision']:.4f}")
    print(f"  Macro Rec : {test_metrics['recall']:.4f}")
    print(f"  Macro AUC : {test_metrics['auroc']:.4f}")
    print("  Per-class AUROC:")
    for cls, auc in test_metrics["per_class_auroc"].items():
        print(f"    {cls:5s}: {auc:.4f}")

    np.save(os.path.join(CHECKPOINT_DIR, "test_preds.npy"), test_preds)
    np.save(os.path.join(CHECKPOINT_DIR, "test_targs.npy"), test_targs)
    with open(os.path.join(CHECKPOINT_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if k != "per_class_auroc"} | {"per_class_auroc": test_metrics["per_class_auroc"]},
                  f, indent=2)
    print(f"\nArtifacts saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
