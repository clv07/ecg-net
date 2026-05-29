# ecg-net

Deep learning for 12-lead ECG classification on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/). This project investigates whether a CNN + Transformer hybrid can outperform classical machine learning, vanilla CNN, ResNet1D, and pure Transformer baselines on the 5-class diagnostic superclass task (NORM, MI, STTC, CD, HYP).

## Repository structure

```
ecg-net/
├── preprocess.py                 # unified signal preprocessing for all deep models
├── feature_extract_xgb.py        # hand-crafted feature extraction for XGBoost
├── baseline_xgboost.py           # Baseline 1: XGBoost + hand-crafted features
├── baseline_cnn.py               # Baseline 2: Vanilla 1D CNN
├── baseline_transformer.py       # Baseline 3: Pure Transformer (ViT-style)
├── baseline_resnet.py            # Baseline 4: ResNet1D
├── proposed_model.py             # Proposed v1: dual-branch (lead-attn-static + CNN+Transformer)
├── proposed_model_v2.py          # Proposed v2: per-lead CNN + spatial lead attention + temporal transformer
├── proposed_model_v3.py          # Proposed v3: multi-scale CNN + single transformer
└── ptb-xl/                       # PTB-XL dataset (download separately from PhysioNet)
```

After running each script, results are saved to `./checkpoint_<model_name>/` (best/last weights, history JSON, test predictions, test metrics).

## Setup

```bash
# Required: numpy < 2.0 for compatibility with system pyarrow/pandas on DSMLP
pip install --user "numpy<2"

# Core dependencies
pip install --user wfdb scipy scikit-learn xgboost neurokit2 tqdm pandas pyarrow

# PyTorch (CUDA 12.1 build for GTX 1080 Ti / Ampere)
pip install --user torch --index-url https://download.pytorch.org/whl/cu121
```

Download PTB-XL v1.0.3 from PhysioNet and place it at `./ptb-xl/`. The directory should contain `ptbxl_database.csv`, `scp_statements.csv`, and the `records100/` folder.

## How to reproduce

The four baselines and three proposed variants share a single preprocessed dataset. Run preprocessing once, then train each model independently.

```bash
# 1. Unified preprocessing (~10 minutes one-time cost)
python preprocess.py

# 2. Hand-crafted feature extraction (only needed for XGBoost, ~5-10 minutes)
python feature_extract_xgb.py

# 3. Train any/all models
python baseline_xgboost.py            # ~3-5 minutes
python baseline_cnn.py                # ~5 minutes on GTX 1080 Ti
python baseline_transformer.py        # ~15 minutes
python baseline_resnet.py             # ~10 minutes
python proposed_model.py              # v1 (~5 minutes)
python proposed_model_v2.py           # v2 (~5 minutes)
python proposed_model_v3.py           # v3 (~5 minutes)
```

---

## Dataset preprocessing

PTB-XL contains 21,799 12-lead, 10-second ECG records from 18,869 patients, sampled at 100 Hz (low-resolution) or 500 Hz (high-resolution). Each record is annotated with diagnostic codes that map to five superclasses: **NORM** (normal), **MI** (myocardial infarction), **STTC** (ST/T changes), **CD** (conduction disturbance), and **HYP** (hypertrophy).

All deep models share the same preprocessing pipeline so the ablation is fair:

```
Raw signals (12, 1000) @ 100 Hz
        ↓
[1] Filter dataset (only samples with valid superclass)
        ↓
[2] Bandpass filter (0.5–40 Hz, zero-phase Butterworth)
    — removes baseline drift (<0.5 Hz) and high-frequency muscle noise (>40 Hz)
        ↓
[3] Train/Val/Test split (PTB-XL recommended folds: 1-8 train, 9 val, 10 test)
        ↓
[4] Per-lead z-score normalization (using training statistics only)
        ↓
[5] Save .npy files for fast loading
        ↓
[6a] Deep models load directly (CNN / ResNet1D / Transformer / Proposed)
[6b] XGBoost extracts hand-crafted features on top of UNNORMALIZED signals
     (amplitude features like Sokolow-Lyon require absolute mV values)
```

Final dataset sizes: train 17,084 · val 2,146 · test 2,158.

### Hand-crafted features for XGBoost (179 features total)

| Group | Features | Count | Clinical relevance |
|---|---|---|---|
| HRV | hr_mean, rr_mean, rr_sdnn, rr_rmssd, n_beats | 5 | Heart rate and rhythm variability |
| Intervals | PR, QRS duration, QT | 3 | Conduction abnormalities (CD class) |
| Per-lead stats (×12) | mean, std, skew, kurt, rms, ptp | 72 | Signal morphology |
| Per-lead spectral (×12) | 4 frequency-band power ratios | 48 | Frequency-domain features |
| Per-lead amplitude (×12) | r_amp, r_amp_max, st_lvl, t_amp | 48 | MI (ST), HYP (R amplitude) |
| Cross-lead | Sokolow-Lyon, Cornell, QRS axis | 3 | LVH diagnostic criteria |

---

## Models

### Baseline 1 — XGBoost (classical ML reference)

One-vs-Rest XGBoost (5 independent binary classifiers) trained on the 179 hand-crafted features. Per-class `scale_pos_weight` for class imbalance, early stopping at 30 rounds, per-class threshold tuning on validation set.

```
Hand-crafted features (N, 179)
    ↓ StandardScaler (fit on train)
    ↓ XGBoost binary classifier × 5 (one per class)
    ↓ Per-class threshold tuning on val
    ↓ Predictions (N, 5)
```

Hyperparameters: `n_estimators=500, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, tree_method=hist`.

---

### Baseline 2 — Vanilla 1D CNN

```
Input: (B, 12, 1000)
    ↓ Conv1d(12→32, k=7) + BN + ReLU + MaxPool(2)
(B, 32, 500)
    ↓ Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)
(B, 64, 250)
    ↓ Conv1d(64→128, k=5) + BN + ReLU + MaxPool(2)
(B, 128, 125)
    ↓ Conv1d(128→256, k=3) + BN + ReLU + MaxPool(2)
(B, 256, 62)
    ↓ Global Average Pool
(B, 256)
    ↓ Dropout → Linear(256→128) → ReLU → Dropout → Linear(128→5)
(B, 5)
```

**Parameters: ~270k.**

| Design choice | Rationale |
|---|---|
| No residual connections | ResNet1D is a separate baseline; keep the contrast clean |
| No attention | Transformer/Proposed is the upgrade; baseline must be plain CNN |
| Kernel size 7 → 5 → 5 → 3 | Wide early kernels capture QRS shape; small late kernels extract abstract features |
| Channels 32 → 64 → 128 → 256 | Classic doubling pattern |
| Global Average Pool | Fewer parameters and more robust than Flatten |
| AdamW + weight decay | Mild regularization |
| Cosine LR + grad clip | Training stability |
| Early stopping (patience=7) | Prevents over-training |

---

### Baseline 3 — Pure Transformer (ViT-style patch embedding)

```
Input: (B, 12, 1000)
    ↓ Reshape into 100 patches of (12, 10)
(B, 100, 120)              ← each patch flattened: 12 leads × 10 timesteps = 120
    ↓ Linear(120 → 128)
(B, 100, 128)
    ↓ Prepend [CLS] token
(B, 101, 128)
    ↓ + Positional Embedding
    ↓ Transformer Encoder × 4 layers (Pre-LN, GELU, nhead=4)
(B, 101, 128)
    ↓ Take [CLS] token output
(B, 128)
    ↓ MLP head: Linear(128→64) → GELU → Dropout → Linear(64→5)
(B, 5)
```

**Parameters: ~567k.** Attention sequence length: 101.

| Design choice | Rationale |
|---|---|
| Patch size = 10 | 100 patches balance attention compute (O(N²)) and temporal resolution (each patch covers 100 ms — about one P-wave or half a QRS) |
| CLS token | ViT-style; a dedicated token aggregates global information, more flexible than mean pooling |
| Pre-LN (`norm_first=True`) | Far more stable than Post-LN when training Transformers from scratch on small datasets |
| Warmup (4 epochs) + cosine decay | Transformers almost always need warmup; large gradients early on can break the attention matrix |
| LR = 3e-4 | Smaller than CNN's 1e-3; Transformers are more LR-sensitive |
| Augmentation: noise / time shift / lead dropout | Transformers lack CNN's inductive bias, so they benefit more from data diversity |
| Dropout = 0.2 | Higher than CNN baseline to combat attention overfitting |
| 4 layers | More depth than the proposed model's 2-3 layers since Transformer here is the *only* feature extractor |
| Epochs = 40 | Transformers converge more slowly than CNNs |

---

### Baseline 4 — ResNet1D

```
Input: (B, 12, 1000)
    ↓ Stem: Conv1d(12→64, k=15, s=2) + BN + ReLU + MaxPool(2)
(B, 64, 250)
    ↓ Stage 1: 2× BasicBlock(64, stride=1)        — same resolution
(B, 64, 250)
    ↓ Stage 2: BasicBlock(64→128, s=2) + BasicBlock(128)
(B, 128, 125)
    ↓ Stage 3: BasicBlock(128→256, s=2) + BasicBlock(256)
(B, 256, 63)
    ↓ Stage 4: BasicBlock(256→512, s=2) + BasicBlock(512)
(B, 512, 32)
    ↓ Global Average Pool + Dropout
(B, 512)
    ↓ Linear(512 → 5)
(B, 5)
```

**Parameters: ~3.86M.**

| Design choice | Rationale |
|---|---|
| ResNet-18 depth (2-2-2-2 blocks) | Matches xresnet1d in Strodthoff et al. 2021; deeper ResNet-50/101 tends to overfit on PTB-XL |
| BasicBlock (not Bottleneck) | Parameter-efficient at this depth; standard in ResNet-18/34 |
| Stem kernel = 15 | Covers ~150 ms — a full QRS complex; standard for ECG ResNet1D |
| Stem stride=2 + MaxPool=2 | Aggressive early downsampling (4×) to reduce downstream compute |
| Channels: 64 → 128 → 256 → 512 | Standard ResNet channel progression |
| Kaiming init | Standard for ReLU networks; BN weights initialized to 1, bias to 0 |
| `bias=False` in Conv layers | Redundant when followed by BN |
| Cosine LR + AdamW | Same as Vanilla CNN for fair comparison |
| Same augmentation as Pure Transformer | Fair comparison |

---

### Proposed v1 — Dual-branch CNN + Lead Attention (failed)

Two parallel branches: (a) a **lead-attention branch** that compresses each lead's 1000 timesteps into a single static embedding then applies self-attention across the 12 lead tokens, and (b) a **multi-scale CNN + temporal transformer branch**. The two branch outputs are concatenated and fed to a classifier. Trained with `pos_weight`-weighted BCE.

**Parameters: ~2.5M.**

This variant underperformed Vanilla CNN. Two main reasons:

1. Compressing each lead into a single static embedding destroyed the temporal information needed for waveform-based diagnosis (R-peak amplitude, ST level).
2. The aggressive `pos_weight` distorted the loss landscape and made the model over-predict positives across all classes.

---

### Proposed v2 — Per-lead CNN + Spatial Lead Attention + Temporal Transformer

The redesigned proposed model. Lead attention is applied at **every downsampled time step**, preserving waveform timing throughout.

```
Input: (B, 12, 1000)
    ↓ Per-lead grouped CNN (groups=12)            ← 12 leads processed independently in parallel
       1000 → 500 → 250 → 62 timesteps
(B, 12, 32, 62)
    ↓ Spatial Lead Attention at each timestep
       reshape: (B*62, 12, 32) → self-attention across 12 leads
       reshape back
(B, 12, 32, 62)
    ↓ Merge leads: Linear(12*32 → 192) per timestep
(B, 62, 192)
    ↓ + Positional Embedding
    ↓ Temporal Transformer × 3 layers (Pre-LN, GELU, nhead=4)
(B, 62, 192)
    ↓ Global Average Pool over time
(B, 192)
    ↓ MLP → (B, 5)
```

**Parameters: ~1.5M.**

| Design choice | Rationale |
|---|---|
| Grouped Conv1d (groups=12) | Each lead processed independently in parallel — no premature lead mixing, GPU-efficient |
| Lead attention broadcast over time | Each time step independently attends across 12 leads; preserves waveform timing while learning cross-lead relationships (e.g., Sokolow-Lyon for HYP, V1↔V6 morphology for LBBB) |
| Per-lead positional embedding for lead attention | The model can learn that V1 ≠ I, etc. |
| Pre-LN Transformer | Same stability rationale as Pure Transformer baseline |
| Standard BCE (no `pos_weight`) | Lesson learned from v1 |
| Per-class threshold tuning on val | Free improvement over default 0.5 threshold |
| No amplitude scaling augmentation | Would destroy the lead-amplitude relations the lead attention is trying to learn |
| No lead dropout augmentation | Would break Sokolow-Lyon-style cross-lead patterns |

---

### Proposed v3 — Multi-scale CNN + Single Transformer (simplified)

A simpler variant exploring whether multi-scale convolutions (kernel sizes 5/15/31 in parallel) plus a temporal Transformer can match v2 without the lead-attention complexity.

```
Input: (B, 12, 1000)
    ↓ MultiScaleConvBlock(12→48) [k=5,15,31] + MaxPool
(B, 48, 250)
    ↓ MultiScaleConvBlock(48→96) + MaxPool
(B, 96, 62)
    ↓ MultiScaleConvBlock(96→144)
(B, 144, 62)
    ↓ + Positional Embedding
    ↓ Transformer Encoder × 3 (Pre-LN, GELU, nhead=4)
    ↓ Global Average Pool + MLP
(B, 5)
```

**Parameters: ~800k.** Result: worse than v2 — lead attention does provide measurable value.

---

## Results

PTB-XL diagnostic superclass classification. Multi-label, 5 classes. Macro-averaged metrics on the held-out test set (PTB-XL fold 10). Per-class thresholds tuned on validation set (fold 9).

| Model | Params | Macro F1 | Macro AUROC | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 5×~400 trees | 0.7070 | 0.9015 | 0.9238 | 0.8720 | 0.9158 | 0.8982 | 0.8976 |
| Pure Transformer | 567k | 0.6854 | 0.9078 | 0.9362 | 0.9057 | 0.9276 | 0.8820 | 0.8876 |
| Vanilla CNN | 270k | **0.7440** | **0.9261** | **0.9459** | **0.9262** | **0.9384** | 0.9188 | 0.9011 |
| ResNet1D | 3.86M | 0.7302 | 0.9231 | 0.9440 | 0.9145 | 0.9334 | 0.9201 | **0.9037** |
| Proposed v1 | 2.5M | 0.7255 | 0.9129 | 0.9383 | 0.8957 | 0.9263 | 0.9024 | 0.9019 |
| **Proposed v2** | 1.5M | **0.7444** | 0.9233 | 0.9461 | 0.9201 | 0.9359 | **0.9215** | 0.8929 |
| Proposed v3 | 800k | 0.7173 | 0.9095 | 0.9354 | 0.8886 | 0.9252 | 0.9055 | 0.8926 |

**Bold** = best per column.

### Observations

1. **Vanilla CNN is the strongest overall model** by macro AUROC (0.9261). At 270k parameters, it outperforms both the 3.86M ResNet1D and all three proposed variants on macro AUROC. This suggests PTB-XL's superclass task may be close to a CNN ceiling at this data scale (~17k training samples).
2. **Proposed v2 ties Vanilla CNN on macro F1** (0.7444 vs 0.7440) and **achieves the best CD AUROC across all models** (0.9215). The spatial lead attention provides a measurable benefit for conduction disturbance, which depends on cross-lead QRS morphology.
3. **Pure Transformer is the weakest deep model** (0.9078 AUROC, 0.6854 F1), confirming that CNN inductive bias matters when training from scratch on a small dataset.
4. **XGBoost is surprisingly competitive** on HYP (0.8976) — only 0.003 behind Vanilla CNN. Hand-crafted features like Sokolow-Lyon are well-aligned with the clinical definition of hypertrophy.
5. **The hardest classes are CD and HYP** across every model. Both depend on global/cross-lead reasoning that simple feed-forward CNNs struggle with.

### XGBoost top features (averaged importance across the 5 OvR classifiers)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | V6_t_amp | 0.0345 |
| 2 | I_t_amp | 0.0281 |
| 3 | II_skew | 0.0262 |
| 4 | qt_interval | 0.0170 |
| 5 | V6_rms | 0.0152 |
| 6 | V4_r_amp | 0.0146 |
| 7 | II_r_amp | 0.0141 |
| 8 | sokolow_lyon | 0.0136 |
| 9 | V6_std | 0.0130 |
| 10 | V5_t_amp | 0.0128 |

T-wave amplitudes (V6, I, V5), interval features (QT), and the Sokolow-Lyon index all rank high — these are exactly the features cardiologists use clinically.

---

## References

- Wagner, P. et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*.
- Strodthoff, N. et al. (2021). Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. *IEEE J. Biomed. Health Inform.*
- Hannun, A. Y. et al. (2019). Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*.
