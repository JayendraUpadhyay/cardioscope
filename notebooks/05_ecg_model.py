# =============================================================================
# CardioScope — Notebook 05: ECG Deep Learning Model (1D CNN Autoencoder)
# =============================================================================
# Hack4Health / Byte2Beat — Seed: 42 (reproducibility requirement)
#
# APPROACH: Unsupervised anomaly detection using a 1D Convolutional Autoencoder.
# Reconstruction error = anomaly score. Higher error = more unusual ECG pattern.
# No ground-truth labels exist for this dataset (see Notebook 04 for details).
# =============================================================================

import random
import numpy as np
import pandas as pd
import os
import sys
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR    = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_LEN  = 2000
STRIDE      = 1000
LATENT_DIM  = 64
# Larger batches preserve the same optimization objective while making the
# required 50-epoch CPU run feasible on a development machine.
BATCH_SIZE  = 128
EPOCHS      = 50
LR          = 1e-3

print("=" * 70)
print("CardioScope — 05 ECG Autoencoder Model")
print("=" * 70)
print(f"  Device:       {DEVICE}")
print(f"  Window size:  {WINDOW_LEN} samples")
print(f"  Stride:       {STRIDE} samples")
print(f"  Latent dim:   {LATENT_DIM}")
print(f"  Batch size:   {BATCH_SIZE}")
print(f"  Epochs:       {EPOCHS}")

# =============================================================================
# 1. LOAD SIGNALS
# =============================================================================
print("\n[1/9] Loading preprocessed ECG signals...")

signals_path = os.path.join(DATA_DIR, 'ecg_signals_usable.npy')

if os.path.exists(signals_path):
    signals = np.load(signals_path)
    print(f"  Loaded from cache: {signals.shape}")
else:
    # Re-extract from raw CSV
    print("  Cache not found — extracting from ecg_timeseries.csv...")

    def clean_val(v):
        if isinstance(v, str):
            if '.' in v:
                parts = v.split('.')
                if len(parts) > 2:
                    v = '.'.join(parts[:-1])
                elif len(parts) == 2 and parts[1].isdigit():
                    v = parts[0]
            try:
                return float(v)
            except ValueError:
                return np.nan
        return float(v)

    ecg_path = os.path.join(DATA_DIR, 'ecg_timeseries.csv')
    signals  = []
    row_idx  = 0

    for chunk in pd.read_csv(ecg_path, chunksize=50):
        for _, row in chunk.iterrows():
            if row_idx >= 187:
                break
            fv   = clean_val(row.iloc[0])
            sig  = np.concatenate(([fv],
                                   row.iloc[1:21892].values.astype(float),
                                   row.iloc[43783:109445].values.astype(float)))
            signals.append(sig)
            row_idx += 1
        if row_idx >= 187:
            break

    signals = np.array(signals)
    np.save(signals_path, signals)
    print(f"  Saved: {signals_path}")

print(f"  Signals shape: {signals.shape}  (recordings × samples)")

# =============================================================================
# 2. ROW-WISE NORMALIZATION
# =============================================================================
print("\n[2/9] Normalizing each recording to [0, 1]...")
for i in range(len(signals)):
    lo, hi = signals[i].min(), signals[i].max()
    if hi > lo:
        signals[i] = (signals[i] - lo) / (hi - lo)

# =============================================================================
# 3. WINDOWING
# =============================================================================
print("\n[3/9] Windowing signals into fixed-length segments...")
windows = []
source_rows = []

for rec_idx, sig in enumerate(signals):
    start = 0
    while start + WINDOW_LEN <= len(sig):
        windows.append(sig[start:start + WINDOW_LEN])
        source_rows.append(rec_idx)
        start += STRIDE

windows      = np.array(windows, dtype=np.float32)
source_rows  = np.array(source_rows)
print(f"  Total windows:   {len(windows):,}")
print(f"  Window shape:    {windows.shape}")
print(f"  Windows per rec: {len(windows) / len(signals):.1f} (avg)")

# =============================================================================
# 4. DATASET & DATALOADERS
# =============================================================================

class ECGWindowDataset(Dataset):
    def __init__(self, windows):
        # shape: (N, 1, window_len)
        self.windows = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

# 80/20 split by recording (not by window).  A random window split would put
# adjacent windows from the same ECG in both sets and overstate reconstruction
# performance. There is deliberately no supervised test metric: no labels exist.
rng = np.random.default_rng(SEED)
recording_ids = rng.permutation(len(signals))
n_val_recordings = max(1, int(len(recording_ids) * 0.20))
val_recordings = recording_ids[:n_val_recordings]
train_recordings = recording_ids[n_val_recordings:]
val_idx = np.flatnonzero(np.isin(source_rows, val_recordings))
train_idx = np.flatnonzero(np.isin(source_rows, train_recordings))

print(f"\n[4/9] Recording-level split: {len(train_recordings)} train / "
      f"{len(val_recordings)} validation recordings")
print(f"         Windows: {len(train_idx):,} train / {len(val_idx):,} validation")

train_ds = ECGWindowDataset(windows[train_idx])
val_ds   = ECGWindowDataset(windows[val_idx])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=True if DEVICE.type == 'cuda' else False)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =============================================================================
# 5. MODEL ARCHITECTURE: 1D CNN AUTOENCODER
# =============================================================================

class ECGAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for ECG anomaly detection.
    Encoder compresses a 2000-sample window to a 64-dim latent vector.
    Decoder reconstructs the original signal. High reconstruction error
    indicates an unusual/anomalous pattern.
    """
    def __init__(self, window_len: int = 2000, latent_dim: int = 64):
        super().__init__()

        # Encoder: 2000 -> 1000 -> 500 -> 250 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16,  kernel_size=7, stride=2, padding=3),   # -> 1000
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),   # -> 500
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),   # -> 250
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(latent_dim),                         # -> latent_dim
        )

        # Decoder: latent_dim -> ... -> 2000
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=5, stride=2,
                               padding=2, output_padding=1),          # -> 128
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=4,
                               padding=2, output_padding=1),          # -> 512
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=4,
                               padding=2, output_padding=1),          # -> 2048
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),               # -> 2048
            nn.Sigmoid(),
        )

        self.window_len = window_len

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        out = self.decoder(z)
        # Trim or pad to exactly window_len
        out = out[:, :, :self.window_len]
        return out

    def forward(self, x):
        z   = self.encode(x)
        out = self.decode(z)
        return out


model = ECGAutoencoder(window_len=WINDOW_LEN, latent_dim=LATENT_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[5/9] Model Architecture:")
print(f"  Trainable parameters: {n_params:,}")

# =============================================================================
# 6. TRAINING LOOP
# =============================================================================
print(f"\n[6/9] Training autoencoder for {EPOCHS} epochs...")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=0.5, patience=5)

best_val_loss = float('inf')
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    epoch_train_loss = 0.0
    for batch in train_dl:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon  = model(batch)
        loss   = criterion(recon, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item() * len(batch)

    epoch_train_loss /= len(train_ds)

    # --- Validate ---
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch in val_dl:
            batch     = batch.to(DEVICE)
            recon     = model(batch)
            loss      = criterion(recon, batch)
            epoch_val_loss += loss.item() * len(batch)

    epoch_val_loss /= len(val_ds)
    scheduler.step(epoch_val_loss)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'ecg_autoencoder.pt'))

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={epoch_train_loss:.6f}  "
              f"val_loss={epoch_val_loss:.6f}"
              + (" [BEST]" if epoch_val_loss == best_val_loss else ""))

print(f"\n  Best val loss: {best_val_loss:.6f}")

# =============================================================================
# 7. LOSS CURVE PLOT
# =============================================================================
print("\n[7/9] Plotting training curves...")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', color='steelblue',  linewidth=2)
ax.plot(range(1, EPOCHS + 1), val_losses,   label='Val Loss',   color='tomato',     linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('ECG Autoencoder Training Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ecg_training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/ecg_training_loss.png")

# =============================================================================
# 8. ANOMALY SCORES
# =============================================================================
print("\n[8/9] Computing per-recording anomaly scores...")

# Load best model
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ecg_autoencoder.pt'),
                                  map_location=DEVICE))
model.eval()

anomaly_scores = []
for rec_idx in range(len(signals)):
    rec_windows = windows[source_rows == rec_idx]
    if len(rec_windows) == 0:
        anomaly_scores.append(np.nan)
        continue

    ds   = ECGWindowDataset(rec_windows)
    dl   = DataLoader(ds, batch_size=32, shuffle=False)
    mses = []

    with torch.no_grad():
        for batch in dl:
            batch = batch.to(DEVICE)
            recon = model(batch)
            mse   = ((recon - batch) ** 2).mean(dim=[1, 2])
            mses.extend(mse.cpu().numpy().tolist())

    anomaly_scores.append(np.mean(mses))

anomaly_scores = np.array(anomaly_scores)
np.save(os.path.join(MODELS_DIR, 'ecg_anomaly_scores.npy'), anomaly_scores)
with open(os.path.join(MODELS_DIR, 'ecg_model_metadata.json'), 'w') as f:
    import json
    json.dump({
        'seed': SEED,
        'usable_recordings': int(len(signals)),
        'window_length': WINDOW_LEN,
        'stride': STRIDE,
        'validation_split': 'recording-level 80/20',
        'best_validation_mse': float(best_val_loss),
        'label_status': 'No ground-truth labels; reconstruction error is an anomaly proxy.'
    }, f, indent=2)
print(f"  Saved: models/ecg_anomaly_scores.npy")
print(f"  Saved: models/ecg_model_metadata.json")

scores_valid = anomaly_scores[~np.isnan(anomaly_scores)]
print(f"  Mean anomaly score:   {scores_valid.mean():.6f}")
print(f"  Median anomaly score: {np.median(scores_valid):.6f}")
print(f"  Min anomaly score:    {scores_valid.min():.6f}")
print(f"  Max anomaly score:    {scores_valid.max():.6f}")

# Histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(scores_valid, bins=30, color='steelblue', alpha=0.85, edgecolor='white')
threshold = np.percentile(scores_valid, 85)
ax.axvline(threshold, color='tomato', linewidth=2, linestyle='--',
           label=f'85th percentile = {threshold:.4f}')
ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('ECG Anomaly Score Distribution (187 recordings)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ecg_anomaly_score_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/ecg_anomaly_score_distribution.png")

# =============================================================================
# 9. SALIENCY MAP (Gradient-based)
# =============================================================================
print("\n[9/9] Computing saliency map for a sample window...")

sample_sig = torch.tensor(windows[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
sample_sig.requires_grad_(True)

model.eval()
recon      = model(sample_sig)
recon_loss = criterion(recon, sample_sig)
recon_loss.backward()

saliency   = sample_sig.grad.abs().squeeze().cpu().numpy()
signal_np  = windows[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('ECG Saliency Map — Gradient of Reconstruction Error w.r.t. Input',
             fontsize=13, fontweight='bold')

# Plot 1: original signal
ax1.plot(signal_np[:3000], linewidth=0.8, color='steelblue')
ax1.set_ylabel('Normalized Amplitude', fontsize=10)
ax1.set_title('Original ECG Signal (first 3000 samples)', fontsize=11)
ax1.grid(True, alpha=0.2)

# Plot 2: saliency overlay
ax2.plot(signal_np[:3000], linewidth=0.8, color='steelblue', alpha=0.5, label='ECG')
ax2_twin = ax2.twinx()
plot_len = min(3000, len(saliency))
ax2_twin.fill_between(range(plot_len), saliency[:plot_len], alpha=0.4, color='tomato', label='Saliency')
ax2.set_xlabel('Sample Index', fontsize=10)
ax2.set_ylabel('Normalized Amplitude', fontsize=10)
ax2_twin.set_ylabel('|Gradient| (Saliency)', fontsize=10, color='tomato')
ax2.set_title('Saliency Overlay (red = model focused here)', fontsize=11)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ecg_saliency_sample.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/ecg_saliency_sample.png")

print("\n" + "=" * 70)
print("ECG MODEL TRAINING COMPLETE")
print(f"  Best val MSE:       {best_val_loss:.6f}")
print(f"  Anomaly score mean: {scores_valid.mean():.6f}")
print(f"  Model saved:        models/ecg_autoencoder.pt")
print(f"\n  LIMITATION REMINDER:")
print(f"  Reconstruction error is a proxy for rhythm anomaly — NOT a clinical")
print(f"  label. No ground-truth exists for this dataset. Do not interpret")
print(f"  high anomaly scores as confirmed pathology.")
print("\n  Next: Notebook 06 — ONNX Export")
print("=" * 70)
