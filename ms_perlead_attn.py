import os
import sys
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
PREPROCESSED_DIR = "./preprocessed"

CHECKPOINT_DIR = "./checkpoint_proposed_v4"
BEST_MDL       = os.path.join(CHECKPOINT_DIR, "best.pt")
LAST_MDL       = os.path.join(CHECKPOINT_DIR, "last.pt")
HISTORY_PATH   = os.path.join(CHECKPOINT_DIR, "history.json")

LEAD_DIM         = 32
D_MODEL          = 192
NHEAD            = 4
NUM_LAYERS       = 3
DIM_FEEDFORWARD  = 384
LEAD_NHEAD       = 4
LEAD_LAYERS      = 1
DROPOUT          = 0.15

BATCH_SIZE       = 64
EPOCHS           = 50
LEARNING_RATE    = 5e-4
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 4
EARLY_STOP_PATIENCE = 12
GRAD_CLIP        = 1.0

MS_KERNELS = (5, 15, 31)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pin_memory_enabled(device):
    return device.type == "cuda"


def dataloader_num_workers():
    return 0 if sys.platform == "darwin" else 2


def load_preprocessed(out_dir=None):
    out_dir = os.path.abspath(
        out_dir or os.environ.get("PREPROCESSED_DIR", PREPROCESSED_DIR)
    )
    splits = {}
    for name in ["train", "val", "test"]:
        x_path = os.path.join(out_dir, f"{name}_X.npy")
        y_path = os.path.join(out_dir, f"{name}_y.npy")
        if not os.path.isfile(x_path) or not os.path.isfile(y_path):
            raise FileNotFoundError(
                f"Missing {name}_X.npy or {name}_y.npy in {out_dir}"
            )
        splits[name] = (np.load(x_path), np.load(y_path))
    return (
        splits["train"][0], splits["train"][1],
        splits["val"][0], splits["val"][1],
        splits["test"][0], splits["test"][1],
    )


class PTBXLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, x):
        x = x.copy()
        if np.random.rand() < 0.5:
            x += np.random.randn(*x.shape).astype(np.float32) * 0.02
        if np.random.rand() < 0.5:
            shift = np.random.randint(-50, 50)
            x = np.roll(x, shift, axis=-1)
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


class MultiScaleGroupedConv(nn.Module):
    def __init__(self, num_leads, in_per_lead, out_per_lead,
                 kernels=MS_KERNELS, stride=1):
        super().__init__()
        n_br = len(kernels)
        base = out_per_lead // n_br
        widths = [base] * (n_br - 1) + [out_per_lead - base * (n_br - 1)]

        self.branches = nn.ModuleList()
        for k, w in zip(kernels, widths):
            in_ch = num_leads * in_per_lead
            out_ch = num_leads * w
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride,
                          padding=k // 2, groups=num_leads, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class MultiScalePerLeadCNN(nn.Module):
    def __init__(self, num_leads=12, lead_dim=32):
        super().__init__()
        self.num_leads = num_leads
        self.lead_dim = lead_dim
        c1, c2 = 8, 16

        self.block1 = MultiScaleGroupedConv(num_leads, 1, c1, stride=2)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = MultiScaleGroupedConv(num_leads, c1, c2, stride=2)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = MultiScaleGroupedConv(num_leads, c2, lead_dim, stride=1)

    def forward(self, x):
        B, L, _ = x.shape
        h = self.pool1(self.block1(x))
        h = self.pool2(self.block2(h))
        h = self.block3(h)
        T_prime = h.size(-1)
        return h.view(B, L, self.lead_dim, T_prime)


class SpatialLeadAttention(nn.Module):
    def __init__(self, num_leads=12, lead_dim=32, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
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

    def forward(self, x):
        B, L, D, T_prime = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B * T_prime, L, D)
        x = x + self.lead_pos_embed
        x = self.attn(x)
        return x.view(B, T_prime, L, D).permute(0, 2, 3, 1).contiguous()


class ProposedECGModelV4(nn.Module):
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
        self.lead_dim = lead_dim

        self.per_lead_cnn = MultiScalePerLeadCNN(num_leads=num_leads, lead_dim=lead_dim)
        self.lead_attention = SpatialLeadAttention(
            num_leads=num_leads,
            lead_dim=lead_dim,
            nhead=lead_nhead,
            num_layers=lead_layers,
            dropout=dropout,
        )
        self.lead_merge = nn.Sequential(
            nn.Linear(num_leads * lead_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
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
        h = self.per_lead_cnn(x)
        h = self.lead_attention(h)
        B, L, D, T_prime = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(B, T_prime, L * D)
        h = self.lead_merge(h)
        h = h + self.time_pos_embed[:, :T_prime, :]
        h = self.temporal_transformer(h)
        h = self.temporal_norm(h)
        return self.classifier(h.mean(dim=1))


def make_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def tune_thresholds(model, loader, device, num_classes):
    model.eval()
    preds_all, targs_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds_all.append(torch.sigmoid(model(x)).cpu().numpy())
            targs_all.append(y.numpy())
    preds = np.vstack(preds_all)
    targs = np.vstack(targs_all)

    thresholds = np.zeros(num_classes)
    for c in range(num_classes):
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            f1 = f1_score(targs[:, c], (preds[:, c] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    preds_all, targs_all = [], []
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
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


def evaluate(model, loader, criterion, device, thresholds=None):
    model.eval()
    running = 0.0
    preds_all, targs_all = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval ", leave=False):
            x, y = x.to(device), y.to(device)
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
        "loss": running / len(loader),
        "f1": f1_score(targs, binary, average="macro", zero_division=0),
        "precision": precision_score(targs, binary, average="macro", zero_division=0),
        "recall": recall_score(targs, binary, average="macro", zero_division=0),
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


def main():
    device = get_device()
    print(f"Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed()
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    kw = dict(batch_size=BATCH_SIZE, num_workers=dataloader_num_workers(),
              pin_memory=pin_memory_enabled(device))
    train_loader = DataLoader(PTBXLDataset(X_train, y_train, augment=True),
                              shuffle=True, **kw)
    val_loader = DataLoader(PTBXLDataset(X_val, y_val), shuffle=False, **kw)
    test_loader = DataLoader(PTBXLDataset(X_test, y_test), shuffle=False, **kw)

    model = ProposedECGModelV4(num_classes=len(SUPERCLASSES)).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": [],
               "val_precision": [], "val_recall": [], "val_auroc": [], "lr": []}
    best_auroc, patience = 0.0, 0

    for epoch in range(EPOCHS):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
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
        print(f"  Val    F1={val_metrics['f1']:.4f}  AUROC={val_metrics['auroc']:.4f}")

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": val_metrics["auroc"]}, LAST_MDL)

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auroc": val_metrics["auroc"]}, BEST_MDL)
            print(f"  Best saved (val AUROC {best_auroc:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch + 1}")
                break

        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    ckpt = torch.load(BEST_MDL, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    thresholds = tune_thresholds(model, val_loader, device, len(SUPERCLASSES))
    np.save(os.path.join(CHECKPOINT_DIR, "thresholds.npy"), thresholds)

    test_metrics, test_preds, test_targs = evaluate(
        model, test_loader, criterion, device, thresholds=thresholds)

    print("\n" + "=" * 60)
    print("TEST — MS Per-Lead Attn")
    print("=" * 60)
    print(f"  Macro F1  : {test_metrics['f1']:.4f}")
    print(f"  Macro AUC : {test_metrics['auroc']:.4f}")
    for cls, auc in test_metrics["per_class_auroc"].items():
        print(f"    {cls:5s}: {auc:.4f}")

    np.save(os.path.join(CHECKPOINT_DIR, "test_preds.npy"), test_preds)
    np.save(os.path.join(CHECKPOINT_DIR, "test_targs.npy"), test_targs)
    with open(os.path.join(CHECKPOINT_DIR, "test_metrics.json"), "w") as f:
        json.dump({**{k: v for k, v in test_metrics.items() if k != "per_class_auroc"},
                   "per_class_auroc": test_metrics["per_class_auroc"]}, f, indent=2)
    print(f"\nSaved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
