"""
Baseline 2: Vanilla 1D CNN
============================
A straightforward 1D CNN baseline as promised in the project proposal.
Architecture: 4 Conv blocks → global average pool → MLP classifier.

No transformer, no attention, no residual connections — this is the
"shallow deep learning" baseline against which we compare ResNet1D
and our proposed CNN+Transformer model.

Run:
    python baseline_cnn.py
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
CHECKPOINT_DIR = "./checkpoint_cnn"
BEST_MDL       = os.path.join(CHECKPOINT_DIR, "best.pt")
LAST_MDL       = os.path.join(CHECKPOINT_DIR, "last.pt")
HISTORY_PATH   = os.path.join(CHECKPOINT_DIR, "history.json")

BATCH_SIZE     = 64
EPOCHS         = 30
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
EARLY_STOP_PATIENCE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================
class PTBXLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ============================================================
# MODEL — Vanilla 1D CNN
# ============================================================
class VanillaCNN(nn.Module):
    """
    Four conv blocks, each: Conv1d → BatchNorm → ReLU → MaxPool.
    Global average pool over time, then a small MLP head.

    Input shape:  (B, 12, 1000)
    Output shape: (B, num_classes)
    """
    def __init__(self, num_leads=12, num_classes=5, base_channels=32, dropout=0.3):
        super().__init__()

        def block(in_ch, out_ch, k=7, pool=2):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool),
            )

        c1 = base_channels         # 32
        c2 = base_channels * 2     # 64
        c3 = base_channels * 4     # 128
        c4 = base_channels * 8     # 256

        # input (B, 12, 1000)
        self.features = nn.Sequential(
            block(num_leads, c1, k=7, pool=2),   # (B,  32, 500)
            block(c1,        c2, k=5, pool=2),   # (B,  64, 250)
            block(c2,        c3, k=5, pool=2),   # (B, 128, 125)
            block(c3,        c4, k=3, pool=2),   # (B, 256,  62)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)   # (B, 256, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                # (B, 256)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    # per-class AUROC for the report
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

    # Load preprocessed data (shared with all baselines)
    print("Loading preprocessed data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed()
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    train_loader = DataLoader(PTBXLDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(PTBXLDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))
    test_loader  = DataLoader(PTBXLDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=(DEVICE == "cuda"))

    print("\nInitializing model...")
    model = VanillaCNN(num_classes=len(SUPERCLASSES)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_f1": [],
               "val_loss":   [], "val_f1":   [],
               "val_precision": [], "val_recall": [], "val_auroc": []}

    best_f1 = 0.0
    patience = 0

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics, _, _    = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_auroc"].append(val_metrics["auroc"])

        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch + 1:02d}/{EPOCHS}  (lr={lr:.2e})")
        print(f"  Train  Loss={train_loss:.4f}  F1={train_f1:.4f}")
        print(f"  Val    Loss={val_metrics['loss']:.4f}  "
              f"F1={val_metrics['f1']:.4f}  "
              f"Prec={val_metrics['precision']:.4f}  "
              f"Recall={val_metrics['recall']:.4f}  "
              f"AUROC={val_metrics['auroc']:.4f}")

        # checkpoint last
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

        # save history every epoch (so we can plot mid-training)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    # ----- Test on best checkpoint -----
    print("\nEvaluating best model on test set...")
    ckpt = torch.load(BEST_MDL, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_preds, test_targs = evaluate(model, test_loader, criterion)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS — Vanilla 1D CNN")
    print("=" * 60)
    print(f"  Loss      : {test_metrics['loss']:.4f}")
    print(f"  Macro F1  : {test_metrics['f1']:.4f}")
    print(f"  Macro Prec: {test_metrics['precision']:.4f}")
    print(f"  Macro Rec : {test_metrics['recall']:.4f}")
    print(f"  Macro AUC : {test_metrics['auroc']:.4f}")
    print("  Per-class AUROC:")
    for cls, auc in test_metrics["per_class_auroc"].items():
        print(f"    {cls:5s}: {auc:.4f}")

    # persist test outputs for later comparison/plotting
    np.save(os.path.join(CHECKPOINT_DIR, "test_preds.npy"), test_preds)
    np.save(os.path.join(CHECKPOINT_DIR, "test_targs.npy"), test_targs)
    with open(os.path.join(CHECKPOINT_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if k != "per_class_auroc"} | {"per_class_auroc": test_metrics["per_class_auroc"]},
                  f, indent=2)
    print(f"\nArtifacts saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
