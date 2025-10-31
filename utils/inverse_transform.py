import torch
from src.utils.audio_io import overlap_add
from src.utils.windowing import generate_synthesis_window

def inverse_stft(
    pred: torch.Tensor,
    hop_len: int,
    window_len: int,
    device: str = "cpu",
    drop_samples: bool = False,
    drop_samples_len: int | None = None,
    ows_len: int | None = None,
    analysis_window: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reconstruct waveform from complex STFT predictions.

    Args:
        pred (torch.Tensor): (B, F, T) complex STFTs.
        hop_len (int): Hop length in samples.
        window_len (int): Analysis window length in samples.
        device (str): Target device.
    """
    y_ifft = torch.fft.irfft(pred, dim=-2, norm="backward")

    if not drop_samples:
        synth_window = torch.ones(window_len) * (hop_len / window_len)
        synth_window = synth_window.to(device).unsqueeze(0).unsqueeze(-1)
    else:
        y_ifft = y_ifft[:, drop_samples_len:, :]
        synth_window = generate_synthesis_window(ows_len, hop_len, analysis_window)
        synth_window = synth_window.to(device).unsqueeze(0).unsqueeze(-1)

    y_win = y_ifft * synth_window
    y_overlap = overlap_add(y_win, hop_len)
    return y_overlap
