import os
import math
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch import nn
from src.dataset import make_loader
from src.model import TemporalCRN
from src.train_utils import train_one_epoch, evaluate, to_device
from src.utils.inverse_transform import inverse_stft
from src.config import (
    SAMPLE_RATE, WINDOW_LEN, HOP_LEN, OWS_LEN,
    DROP_SAMPLES_LEN, ANALYSIS_WINDOW, SNR_RANGE
)

# ------------------------------
# Experiment setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_path = Path("./experiments")
experiment_no = 1

exp_dir = project_path / f"exp{experiment_no}"
(exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
csv_path = exp_dir / "stats.csv"

with open(csv_path, "w") as f:
    f.write("epoch,train_loss,val_loss\n")

# ------------------------------
# Data loaders
# ------------------------------
DATA_DIR = Path("./data")
SRC_PATH = DATA_DIR / "Source_16kHz"
AUX_PATH = DATA_DIR / "Auxiliary_16kHz"

BATCH_SIZE = 32
DURATION_S = 5.0

from src.config import toUseMetaData  # optional if defined externally

ds_train, dl_train = make_loader(
    src_root=SRC_PATH,
    aux_root=AUX_PATH,
    partition="train",
    duration_sec=DURATION_S,
    snr_list=SNR_RANGE,
    metadata=toUseMetaData,
    target_sr=SAMPLE_RATE,
    batch_size=BATCH_SIZE,
    num_workers=8,
)

ds_dev, dl_dev = make_loader(
    src_root=SRC_PATH,
    aux_root=AUX_PATH,
    partition="dev",
    duration_sec=DURATION_S,
    snr_list=SNR_RANGE,
    metadata=toUseMetaData,
    target_sr=SAMPLE_RATE,
    batch_size=BATCH_SIZE,
    num_workers=8,
)

# ------------------------------
# Model and optimizer
# ------------------------------
model = TemporalCRN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# ------------------------------
# Training loop
# ------------------------------
num_epochs = 1000
grad_clip = 5.0
best_val, best_epoch = float("inf"), 0

for epoch in range(1, num_epochs + 1):
    print(f"\n{'-'*50}\nEpoch {epoch}\n{'-'*50}")

    train_loss = train_one_epoch(
        model=model,
        dataloader=dl_train,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        hop_len=HOP_LEN,
        window_len=WINDOW_LEN,
        drop_samples_len=DROP_SAMPLES_LEN,
        ows_len=OWS_LEN,
        analysis_window=ANALYSIS_WINDOW,
        grad_clip=grad_clip,
    )

    val_loss = evaluate(
        model=model,
        dataloader=dl_dev,
        device=device,
        hop_len=HOP_LEN,
        window_len=WINDOW_LEN,
        drop_samples_len=DROP_SAMPLES_LEN,
        ows_len=OWS_LEN,
        analysis_window=ANALYSIS_WINDOW,
    )

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    with open(csv_path, "a") as f:
        f.write(f"{epoch},{train_loss},{val_loss}\n")

    # Save checkpoint
    ckpt_path = exp_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
    torch.save(model.state_dict(), ckpt_path)

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch

print(f"\nTraining complete. Best validation loss: {best_val:.4f} (epoch {best_epoch})")

# ------------------------------
# Plot training curves
# ------------------------------
df = pd.read_csv(csv_path)
df.plot(x="epoch", y=["train_loss", "val_loss"], title="Training Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ------------------------------
# Testing
# ------------------------------
def compute_snr(est: np.ndarray, gt: np.ndarray, eps=1e-9):
    """Compute signal-to-noise ratio (SNR) in dB."""
    s_pwr = np.sum(gt**2) + eps
    n_pwr = np.sum((gt - est) ** 2) + eps
    return 10 * np.log10(s_pwr / n_pwr)


ds_test, dl_test = make_loader(
    src_root=SRC_PATH,
    aux_root=AUX_PATH,
    partition="test",
    duration_sec=DURATION_S,
    snr_list=SNR_RANGE,
    metadata=toUseMetaData,
    target_sr=SAMPLE_RATE,
    batch_size=BATCH_SIZE,
    num_workers=4,
)

batch = next(iter(dl_test))
batch = to_device(batch, device)

pred_stft, _ = model(batch["mix_stft"])
pred_wave = inverse_stft(
    pred_stft,
    hop_len=HOP_LEN,
    window_len=WINDOW_LEN,
    device=device,
    drop_samples=True,
    drop_samples_len=DROP_SAMPLES_LEN,
    ows_len=OWS_LEN,
    analysis_window=ANALYSIS_WINDOW,
)

pred_wave = pred_wave.cpu().detach().numpy()
clean_wave = batch["src_seq"].cpu().detach().numpy()
noisy_wave = batch["mix_seq"].cpu().detach().numpy()

sample_idx = 0
print(f"Input SNR: {compute_snr(noisy_wave[sample_idx], clean_wave[sample_idx]):.2f} dB")
print(f"Output SNR: {compute_snr(pred_wave[sample_idx], clean_wave[sample_idx][DROP_SAMPLES_LEN:]):.2f} dB")
