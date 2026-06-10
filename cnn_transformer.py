import os
import json
import numpy as np
import torch
import torch.nn as nn

from utils import DEVICE, SUPERCLASSES, model_summary, train_one_epoch, evaluate, tune_thresholds, make_cosine, make_warmup_cosine, save_test_artifacts, print_metrics

class PerLeadCNN(nn.Module):
    """
    Grouped Conv1d with groups=num_leads.
    Output: (B, num_leads, lead_dim, T')
    """
    def __init__(self, num_leads=12, lead_dim=32):
        super().__init__()
        self.num_leads = num_leads
        self.lead_dim  = lead_dim

        c1, c2 = 8, 16
        self.cnn = nn.Sequential(
            # input  : (B, 12, 1000)
            # output : (B, 12 * c1, 500)
            nn.Conv1d(num_leads, num_leads * c1, kernel_size=15,
                      stride=2, padding=7, groups=num_leads, bias=False),
            nn.BatchNorm1d(num_leads * c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                               

            nn.Conv1d(num_leads * c1, num_leads * c2, kernel_size=7,
                      stride=2, padding=3, groups=num_leads, bias=False),
            nn.BatchNorm1d(num_leads * c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

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

class SpatialLeadAttention(nn.Module):
    """
    Apply multi-head self-attention across the 12 leads
    at each downsampled time step independently to learn cross-lead relationships.

    Input:  (B, 12, lead_dim, T')
    Output: (B, 12, lead_dim, T')
    """
    def __init__(self, num_leads=12, lead_dim=32, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.lead_pos_embed = nn.Parameter(torch.randn(1, num_leads, lead_dim) * 0.02)
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
        B, L, D, T_prime = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * T_prime, L, D)
        x = x + self.lead_pos_embed
        x = self.attn(x)
        x = x.view(B, T_prime, L, D).permute(0, 2, 3, 1).contiguous()
        return x


class CnnTransformer(nn.Module):
    def __init__(self, num_leads=12, num_classes=5, lead_dim=32, d_model=192, nhead=4, num_layers=3,
                 dim_feedforward=384, lead_nhead=4, lead_layers=1, dropout=0.15):
        super().__init__()
        self.num_leads = num_leads
        self.lead_dim = lead_dim
        self.d_model = d_model

        # per-lead CNN
        self.per_lead_cnn = PerLeadCNN(num_leads=num_leads, lead_dim=lead_dim)

        # spatial lead attention - cross-lead at each timestep
        self.lead_attention = SpatialLeadAttention(
            num_leads=num_leads,
            lead_dim=lead_dim,
            nhead=lead_nhead,
            num_layers=lead_layers,
            dropout=dropout,
        )

        # merge leads: project (12 * lead_dim) -> d_model at each timestep
        self.lead_merge = nn.Sequential(
            nn.Linear(num_leads * lead_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # temporal transformer
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
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_norm = nn.LayerNorm(d_model)

        # classifier
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

        h = h.mean(dim=1)
        return self.classifier(h)


#######################################
# Training
#######################################
def train_proposed(model, cfg, loaders, title="CNN + Transformer"):
    """
    Train the proposed CNN + Transformer model with early stopping on F1 validation
    metric, then evaluate the best checkpoint on the test
    set with both the default 0.5 threshold and tuned per-class thresholds.
    """
    train_loader, val_loader, test_loader = loaders
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_mdl = os.path.join(cfg["checkpoint_dir"], "best.pt")
    last_mdl = os.path.join(cfg["checkpoint_dir"], "last.pt")
    history_path = os.path.join(cfg["checkpoint_dir"], "history.json")

    select_metric = cfg.get("select_metric", "f1")

    model = model.to(DEVICE)
    model_summary(model, name=title)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg.get("warmup_epochs"):
        scheduler = make_warmup_cosine(optimizer, cfg["warmup_epochs"], cfg["epochs"])
    else:
        scheduler = make_cosine(optimizer, cfg["epochs"])

    history = {k: [] for k in ("train_loss", "train_f1", "val_loss", "val_f1",
                               "val_precision", "val_recall", "val_auroc", "lr")}
    best_score, patience = 0.0, 0

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
        history["lr"].append(lr_now)

        print(f"\nEpoch {epoch + 1:02d}/{cfg['epochs']}  (lr={lr_now:.2e})")
        print(f"  Train  Loss={train_loss:.4f}  F1={train_f1:.4f}")
        print(f"  Val    Loss={val_metrics['loss']:.4f}  F1={val_metrics['f1']:.4f}  "
              f"Prec={val_metrics['precision']:.4f}  Recall={val_metrics['recall']:.4f}  "
              f"AUROC={val_metrics['auroc']:.4f}")
        print("  Val per-class AUROC: " + "  ".join(
            f"{cls}={v:.3f}" for cls, v in val_metrics["per_class_auroc"].items()))

        # save the latest checkpoint
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_metrics["f1"], "val_auroc": val_metrics["auroc"]},
                   last_mdl)

        score = val_metrics[select_metric]
        if score > best_score:
            best_score, patience = score, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        f"val_{select_metric}": best_score}, best_mdl)
            print(f"   checkpoint: best model saved "
                  f"(val {select_metric}: {best_score:.4f})")
        else:
            patience += 1
            print(f"  Patience {patience}/{cfg['patience']}")
            if patience >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    # load the best checkpoint
    print("\nLoading best model for evaluation...")
    ckpt = torch.load(best_mdl, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # tune per-class thresholds on the validation set
    print("\nTuning per-class thresholds on validation set...")
    thresholds = tune_thresholds(model, val_loader, len(SUPERCLASSES))
    for cls, t in zip(SUPERCLASSES, thresholds):
        print(f"    {cls:5s}: {t:.3f}")
    np.save(os.path.join(cfg["checkpoint_dir"], "thresholds.npy"), thresholds)

    # test with the 0.5 threshold
    print("\nTest with threshold 0.5:")
    test_default, _, _ = evaluate(model, test_loader, criterion, thresholds=None)
    print(f"  Macro F1={test_default['f1']:.4f}  "
          f"Prec={test_default['precision']:.4f}  "
          f"Recall={test_default['recall']:.4f}")

    # test with tuned per-class thresholds
    print("\nTest with tuned per-class thresholds:")
    test_metrics, test_preds, test_targs = evaluate(
        model, test_loader, criterion, thresholds=thresholds)
    print_metrics(test_metrics, title=f"FINAL TEST RESULTS — {title}")
    save_test_artifacts(cfg["checkpoint_dir"], test_metrics, test_preds, test_targs)
    print(f"\nArtifacts saved to {cfg['checkpoint_dir']}/")
    return test_metrics