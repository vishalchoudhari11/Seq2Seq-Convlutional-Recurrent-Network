import math
import time
import torch
from typing import Dict, Any
from src.losses import l1_waveform_loss, l1_stft_magnitude_loss
from src.utils.inverse_transform import inverse_stft


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a batch of tensors to the target device."""
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    hop_len: int,
    window_len: int,
    drop_samples_len: int,
    ows_len: int,
    analysis_window: torch.Tensor,
) -> float:
    """
    Evaluate the model over a dataset using waveform + STFT L1 losses.
    """
    model.eval()
    total_loss, num_examples = 0.0, 0

    for batch in dataloader:
        batch = to_device(batch, device)

        pred_stft, _ = model(batch["mix_stft"])  # (B, F, T)
        pred_wave = inverse_stft(
            pred_stft,
            hop_len=hop_len,
            window_len=window_len,
            device=device,
            drop_samples=True,
            drop_samples_len=drop_samples_len,
            ows_len=ows_len,
            analysis_window=analysis_window,
        )

        loss = (
            l1_waveform_loss(batch["src_seq"][:, drop_samples_len:], pred_wave)
            + l1_stft_magnitude_loss(batch["src_stft"], pred_stft)
        )

        total_loss += loss.item() * pred_stft.size(0)
        num_examples += pred_stft.size(0)

    return total_loss / max(num_examples, 1)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    hop_len: int,
    window_len: int,
    drop_samples_len: int,
    ows_len: int,
    analysis_window: torch.Tensor,
    grad_clip: float = 5.0,
) -> float:
    """
    Train the model for one epoch.

    Computes combined L1 waveform + STFT magnitude loss.
    """
    model.train()
    total_loss, num_examples = 0.0, 0
    num_batches = len(dataloader)
    start_time = time.time()

    marks = sorted(
        {
            max(1, math.floor(0.25 * num_batches)),
            max(1, math.floor(0.50 * num_batches)),
            max(1, math.floor(0.75 * num_batches)),
            num_batches,
        }
    )
    next_mark_i, next_mark = 0, marks[0]

    for step, batch in enumerate(dataloader, start=1):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        pred_stft, _ = model(batch["mix_stft"])
        pred_wave = inverse_stft(
            pred_stft,
            hop_len=hop_len,
            window_len=window_len,
            device=device,
            drop_samples=True,
            drop_samples_len=drop_samples_len,
            ows_len=ows_len,
            analysis_window=analysis_window,
        )

        loss = (
            l1_waveform_loss(batch["src_seq"][:, drop_samples_len:], pred_wave)
            + l1_stft_magnitude_loss(batch["src_stft"], pred_stft)
        )

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * pred_stft.size(0)
        num_examples += pred_stft.size(0)

        # Periodic progress logging
        if step == next_mark:
            avg_loss = total_loss / num_examples
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:03d} [{step:>4d}/{num_batches}] "
                f"train_loss={avg_loss:.4f} elapsed={elapsed:.1f}s"
            )
            next_mark_i = min(next_mark_i + 1, len(marks) - 1)
            next_mark = marks[next_mark_i]

    return total_loss / max(num_examples, 1)
