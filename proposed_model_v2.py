"""
Proposed Model v2: Per-lead CNN + Spatial-Temporal Attention
=============================================================
Redesigned from v1 to fix the broken Lead Attention module.

Key change from v1:
    v1 compressed each lead's 1000-timestep signal into a single static
    vector before applying lead attention. This destroyed the temporal
    information needed for waveform-based diagnosis (R-peak amplitude,
    ST level, etc.), explaining why v1 underperformed Vanilla CNN.

    v2 keeps the time axis throughout. Per-lead CNN (grouped conv,
    groups=12) processes each lead independently in parallel, producing
    a tensor of shape (B, 12, d_lead, T'). Then lead attention is
    applied at EACH time step: the 12 lead tokens attend to one
    another at every position along the downsampled time axis. This is
    spatial attention in the lead dimension, broadcast across time.

Architecture:
    Input (B, 12, 1000)
      → Per-lead grouped CNN, downsample to T'=62
        (B, 12, d_lead, 62)
      → For each time step independently:
          Lead self-attention across 12 tokens of dim d_lead
        Still (B, 12, d_lead, 62)
      → Merge leads: linear projection (12*d_lead → d_model)
        (B, d_model, 62)
      → Temporal Transformer encoder
        (B, 62, d_model)
      → Global average pool over time
        (B, d_model)
      → Classifier
        (B, num_classes)

Other changes from v1:
    - NO pos_weight (caused over-prediction of minorities in v1)
    - Simpler training (single LR schedule, removed amplitude scaling
      augmentation since lead-attention is sensitive to absolute amps)
    - Per-class threshold tuning retained (free improvement)

Run:
    python proposed_model_v2.py
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
CHECKPOINT_DIR = "./checkpoint_proposed_v2"
BEST_MDL       = os.path.join(CHECKPOINT_DIR, "best.pt")
LAST_MDL       = os.path.join(CHECKPOINT_DIR, "last.pt")
HISTORY_PATH   = os.path.join(CHECKPOINT_DIR, "history.json")

# Architecture
LEAD_DIM         = 32    # per-lead feature dim after grouped CNN
D_MODEL          = 192   # temporal transformer dim
NHEAD            = 4
NUM_LAYERS       = 3
DIM_FEEDFORWARD  = 384
LEAD_NHEAD       = 4
LEAD_LAYERS      = 1
DROPOUT          = 0.15

# Training
BATCH_SIZE       = 64
EPOCHS           = 50
LEARNING_RATE    = 5e-4
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 4
EARLY_STOP_PATIENCE = 12
GRAD_CLIP        = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# DATASET
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
        # Light Gaussian noise
        if np.random.rand() < 0.5:
            x += np.random.randn(*x.shape).astype(np.float32) * 0.02
        # Time shift
        if np.random.rand() < 0.5:
            shift = np.random.randint(-50, 50)
            x = np.roll(x, shift, axis=-1)
        # NOTE: no amplitude scaling (lead attention learns amplitude relations)
        # NOTE: no lead dropout (would destroy lead-pair relations like Sokolow-Lyon)
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


# ============================================================
# MODULE 1 — Per-lead CNN (grouped conv: 12 independent paths in parallel)
# ============================================================
class PerLeadCNN(nn.Module):
    """
    Grouped Conv1d with groups=num_leads: each lead is processed
    independently but in parallel, no lead mixing.

    Output: (B, num_leads, lead_dim, T')
    """
    def __init__(self, num_leads=12, lead_dim=32):
        super().__init__()
        self.num_leads = num_leads
        self.lead_dim  = lead_dim

        # We want each lead to be processed independently.
        # Equivalently: a Conv1d with in=num_leads*1, out=num_leads*C, groups=num_leads
        # produces num_leads independent channels of width C.
        # We stack 3 such blocks, each lead's intermediate channel widths: 8 → 16 → lead_dim
        c1, c2 = 8, 16
        self.cnn = nn.Sequential(
            # input  : (B, 12,        1000)
            # output : (B, 12 * c1,   500)
            nn.Conv1d(num_leads, num_leads * c1, kernel_size=15,
                      stride=2, padding=7, groups=num_leads, bias=False),
            nn.BatchNorm1d(num_leads * c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                # → 250

            nn.Conv1d(num_leads * c1, num_leads * c2, kernel_size=7,
                      stride=2, padding=3, groups=num_leads, bias=False),
            nn.BatchNorm1d(num_leads * c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                # → 62

            nn.Conv1d(num_leads * c2, num_leads * lead_dim, kernel_size=3,
                      stride=1, padding=1, groups=num_leads, bias=False),
            nn.BatchNorm1d(num_leads * lead_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, num_leads, T)
        B, L, T = x.shape
        h = self.cnn(x)                                    # (B, L*lead_dim, T')
        T_prime = h.size(-1)
        # reshape to expose lead dimension
        h = h.view(B, L, self.lead_dim, T_prime)           # (B, 12, lead_dim, T')
        return h


# ============================================================
# MODULE 2 — Spatial-Temporal Lead Attention
# ============================================================
class SpatialLeadAttention(nn.Module):
    """
    At each downsampled time step independently, apply multi-head
    self-attention across the 12 leads (treated as tokens).

    Input:  (B, 12, lead_dim, T')
    Output: (B, 12, lead_dim, T')

    This lets the model learn cross-lead relationships like Sokolow-Lyon
    (V1 + V5/V6) at every time step, while preserving waveform timing
    information.
    """
    def __init__(self, num_leads=12, lead_dim=32, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        # Learnable per-lead positional embedding (so attention knows which
        # lead is which — model can learn that V1 ≠ I, etc.)
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
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.num_leads = num_leads
        self.lead_dim  = lead_dim

    def forward(self, x):
        # x: (B, 12, lead_dim, T')
        B, L, D, T_prime = x.shape
        # Move time to batch: each timestep is processed independently
        # (B, 12, lead_dim, T') → (B, T', 12, lead_dim) → (B*T', 12, lead_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * T_prime, L, D)
        # Add per-lead positional embedding (broadcasts over batch)
        x = x + self.lead_pos_embed
        # Cross-lead self-attention
        x = self.attn(x)
        # Reshape back: (B*T', 12, D) → (B, T', 12, D) → (B, 12, D, T')
        x = x.view(B, T_prime, L, D).permute(0, 2, 3, 1).contiguous()
        return x


# ============================================================
# MAIN MODEL
# ============================================================
class ProposedECGModelV2(nn.Module):
    def __init__(self,
                 num_leads=12,
                 num_classes=5,
                 lead_dim=LEAD_DIM,
                 d_model=D_MODEL,
                 nhead=NHEAD,
                 num_layers=NUM_LAYERS,
                 dim_feedforward=DIM_FEEDFORWARD,
                 lead_nhead=LEAD_NHEAD,
                 lead_layers=LEAD_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.num_leads = num_leads
        self.lead_dim  = lead_dim
        self.d_model   = d_model

        # (1) Per-lead CNN
        self.per_lead_cnn = PerLeadCNN(num_leads=num_leads, lead_dim=lead_dim)

        # (2) Spatial Lead Attention (cross-lead at each timestep)
        self.lead_attention = SpatialLeadAttention(
            num_leads=num_leads,
            lead_dim=lead_dim,
            nhead=lead_nhead,
            num_layers=lead_layers,
            dropout=dropout,
        )

        # (3) Merge leads: project (12 * lead_dim) → d_model at each timestep
        self.lead_merge = nn.Sequential(
            nn.Linear(num_leads * lead_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # (4) Temporal Transformer
        self.time_pos_embed = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
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

        # (5) Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
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

        # (1) Per-lead CNN → (B, 12, lead_dim, T')
        h = self.per_lead_cnn(x)

        # (2) Spatial lead attention at each timestep → (B, 12, lead_dim, T')
        h = self.lead_attention(h)

        # (3) Merge leads at each timestep:
        #     (B, 12, lead_dim, T') → (B, T', 12*lead_dim) → (B, T', d_model)
        B, L, D, T_prime = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(B, T_prime, L * D)
        h = self.lead_merge(h)                              # (B, T', d_model)

        # (4) Temporal transformer
        h = h + self.time_pos_embed[:, :T_prime, :]
        h = self.temporal_transformer(h)
        h = self.temporal_norm(h)

        # (5) Global average pool over time + classify
        h = h.mean(dim=1)
        return self.classifier(h)


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
# THRESHOLD TUNING
# ============================================================
def tune_thresholds(model, loader, num_classes):
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
    binary = (preds > 0.5).astype(int) if thresholds is None \
             else (preds > thresholds[None, :]).astype(int)

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

    print("\nInitializing proposed model v2...")
    model = ProposedECGModelV2(num_classes=len(SUPERCLASSES)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Standard BCE — no pos_weight (v1 lesson learned)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    history = {"train_loss": [], "train_f1": [],
               "val_loss":   [], "val_f1":   [],
               "val_precision": [], "val_recall": [], "val_auroc": [],
               "lr": []}

    best_auroc = 0.0
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

    # ----- Load best -----
    print("\nLoading best model for evaluation...")
    ckpt = torch.load(BEST_MDL, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # ----- Threshold tuning -----
    print("\nTuning per-class thresholds on validation set...")
    thresholds = tune_thresholds(model, val_loader, len(SUPERCLASSES))
    print("  Tuned thresholds:")
    for cls, t in zip(SUPERCLASSES, thresholds):
        print(f"    {cls:5s}: {t:.3f}")
    np.save(os.path.join(CHECKPOINT_DIR, "thresholds.npy"), thresholds)

    # ----- Test (default + tuned) -----
    print("\nTest with default threshold (0.5):")
    test_default, _, _ = evaluate(model, test_loader, criterion, thresholds=None)
    print(f"  Macro F1={test_default['f1']:.4f}  "
          f"Prec={test_default['precision']:.4f}  "
          f"Recall={test_default['recall']:.4f}")

    print("\nTest with tuned per-class thresholds:")
    test_metrics, test_preds, test_targs = evaluate(model, test_loader, criterion,
                                                     thresholds=thresholds)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS — Proposed Model v2")
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
