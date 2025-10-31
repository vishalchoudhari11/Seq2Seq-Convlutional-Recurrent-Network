import torch

def l1_waveform_loss(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """L1 loss on time-domain sequences."""
    return torch.mean(torch.abs(truth - pred))


def l1_stft_magnitude_loss(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """L1 loss on STFT magnitude difference."""
    return torch.mean(torch.abs(torch.abs(truth) - torch.abs(pred)))
