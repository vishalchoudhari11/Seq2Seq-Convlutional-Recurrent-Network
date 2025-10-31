import torch
import warnings
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def load_sequence(
    path: str,
    start_frame: int,
    num_frames: int,
    src_sr: int,
    target_sr: int
) -> torch.Tensor:
    """
    Load a segment from an audio (or generic sequential) file, resample if needed,
    and return a 1-D torch tensor.

    Args:
        path (str): Path to the source file.
        start_frame (int): Starting frame to read.
        num_frames (int): Number of frames to read.
        src_sr (int): Source sampling rate.
        target_sr (int): Target sampling rate (for resampling).
    
    Returns:
        torch.Tensor: 1-D tensor of the loaded sequence.
    """
    with sf.SoundFile(str(path)) as f:
        f.seek(start_frame)
        seq = f.read(frames=num_frames, dtype='float32')
        sr = f.samplerate

    # Convert to mono if multichannel
    if seq.ndim > 1:
        seq = seq.mean(axis=1)

    if sr != src_sr:
        warnings.warn(f"File sample rate ({sr}) != src_sr ({src_sr}). Updating src_sr.")
        src_sr = sr

    # Resample if needed
    if src_sr != target_sr:
        gcd = np.gcd(src_sr, target_sr)
        up = target_sr // gcd
        down = src_sr // gcd
        seq = resample_poly(seq, up, down)

    return torch.from_numpy(seq), src_sr


def overlap_add(frames: torch.Tensor, hop: int) -> torch.Tensor:
    """
    Perform simple overlap-add reconstruction.

    Args:
        frames (torch.Tensor): Shape (B, L, N) — batch, frame length, number of frames.
        hop (int): Hop size in samples.

    Returns:
        torch.Tensor: Reconstructed signal (B, T).
    """
    B, L, N = frames.shape
    out_len = (N - 1) * hop + L
    output = torch.zeros(B, out_len, device=frames.device, dtype=frames.dtype)
    for i in range(N):
        start = i * hop
        output[:, start:start + L] += frames[:, :, i]
    return output
