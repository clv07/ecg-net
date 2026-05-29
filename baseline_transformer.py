"""
Baseline 3: Pure Transformer (ViT-style patch embedding)
=========================================================
A pure Transformer baseline with NO convolutional feature extractor.
Uses patch embedding (similar to Vision Transformer) to convert the
raw 12-lead ECG into a sequence of tokens.

Architecture:
    Input (B, 12, 1000)
    → reshape into 100 non-overlapping patches of (12, 10)
    → linear projection per patch → 100 tokens of dim d_model
    → prepend learnable CLS token
    → add learnable positional embedding
    → N transformer encoder layers
    → CLS token output → classifier

Purpose of this baseline:
    Establishes whether a pure transformer (without convolutional
    inductive bias) is sufficient for PTB-XL classification. If it
    underperforms the vanilla CNN baseline, this directly justifies
    the CNN frontend in our proposed CNN+Transformer model.

Run:
    python baseline_transformer.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from preprocess import load_preprocessed, SUPERCLASSES

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_DIR = "./checkpoint_transformer"
BEST_MDL       = os.path.join(CHECKPOINT_DIR, "best.pt")
LAST_MDL       = os.path.join(CHECKPOINT_DIR, "last.pt")
HISTORY_PATH   = os.path.join(CHECKPOINT_DIR, "history.json")

# Architecture
PATCH_SIZE     = 10      # 10 timesteps per patch → 100 patches total
D_MODEL        = 128
NHEAD          = 4
NUM_LAYERS     = 4
DIM_FEEDFORWARD = 256
DROPOUT        = 0.2

# Training
BATCH_SIZE     = 64
EPOCHS         = 40       # transformer needs more epochs without CNN inductive bias
LEARNING_RATE  = 3e-4     # smaller LR; transformer is more sensitive
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 4
EARLY_STOP_PATIENCE = 10
GRAD_CLIP      = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# DATASET (with light augmentation)
# ============================================================
class PTBXLDataset(Dataset):
    """
    Transformer is data-hungry without CNN inductive bias, so we enable
    light augmentation on the training set:
      - small Gaussian noise
      - random time shift
      - lead dropout (zero out one random lead with low prob)
    """
    def __init__(self, X, y, augment=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, x):
        # x: (12, T) — copy to avoid mutating cached array
        x = x.copy()

        # 1. Gaussian noise (p=0.5)
        if np.random.rand() < 0.5:
            x += np.random.randn(*x.shape).astype(np.float32) * 0.02

        # 2. Random time shift (p=0.5)  — circular shift within ±50 samples
        if np.random.rand() < 0.5:
            shift = np.random.randint(-50, 50)
            x = np.roll(x, shift, axis=-1)

        # 3. Lead dropout (p=0.2)
        if np.random.rand() < 0.2:
            x[np.random.randint(0, 12)] = 0

        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


# ============================================================
# MODEL — Pure Transformer with patch embedding
# ============================================================
class PureTransformerECG(nn.Module):
    """
    No CNN. Patches of the raw signal are linearly projected into tokens.
    """
    def __init__(self,
                 num_leads=12,
                 seq_len=1000,
                 patch_size=PATCH_SIZE,
                 num_classes=5,
                 d_model=D_MODEL,
                 nhead=NHEAD,
                 num_layers=NUM_LAYERS,
                 dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"

        self.patch_size  = patch_size
        self.num_patches = seq_len // patch_size

        # Each patch = (num_leads × patch_size) flattened
        # so the embedding maps R^(12*10) → R^d_model
        self.patch_embed = nn.Linear(num_leads * patch_size, d_model)

        # CLS token (one learnable vector prepended to the sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional embedding (one per patch + 1 for CLS)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model) * 0.02
        )

        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,        # Pre-LN is more stable for training from scratch
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, 12, 1000)
        B = x.size(0)

        # ----- Patchify -----
        # split last dim into (num_patches, patch_size)
        # (B, 12, num_patches, patch_size)
        x = x.unfold(-1, self.patch_size, self.patch_size)
        # (B, num_patches, 12, patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (B, num_patches, 12 * patch_size)
        x = x.reshape(B, self.num_patches, -1)

        # ----- Linear embed each patch -----
        x = self.patch_embed(x)                         # (B, num_patches, d_model)

        # ----- Prepend CLS token -----
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                  # (B, num_patches+1, d_model)

        # ----- Add positional embedding -----
        x = x + self.pos_embed
        x = self.embed_dropout(x)

        # ----- Transformer encoder -----
        x = self.transformer(x)
        x = self.norm(x)

        # ----- Classify via CLS token -----
        cls_out = x[:, 0]                               # (B, d_model)
        return self.classifier(cls_out)


# ============================================================
# LR SCHEDULER — Linear warmup + cosine decay
# ============================================================
def make_scheduler(optimizer, warmup_epochs, total_epochs):
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    f1 = f1_score(targs, (preds > 0.5).astype(int), average="macro", zero_division=0)
    return running / len(loader), f1


def evaluate(model, loader, criterion):
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
    binary = (preds > 0.5).astype(int)

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

    print("\nInitializing Pure Transformer (no CNN)...")
    model = PureTransformerECG(num_classes=len(SUPERCLASSES)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Patches per sample: {model.num_patches} (patch_size={model.patch_size})")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    history = {"train_loss": [], "train_f1": [],
               "val_loss":   [], "val_f1":   [],
               "val_precision": [], "val_recall": [], "val_auroc": [],
               "lr": []}

    best_f1 = 0.0
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

        # last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_metrics["f1"],
        }, LAST_MDL)

        # best by val F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_metrics["f1"],
            }, BEST_MDL)
            print(f"  ✓ Best model saved (val F1: {best_f1:.4f})")
        else:
            patience += 1
            print(f"  Patience {patience}/{EARLY_STOP_PATIENCE}")
            if patience >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    # ----- Test on best checkpoint -----
    print("\nEvaluating best model on test set...")
    ckpt = torch.load(BEST_MDL, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_preds, test_targs = evaluate(model, test_loader, criterion)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS — Pure Transformer (ViT-style)")
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
