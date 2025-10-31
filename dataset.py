import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
import soundfile as sf

from src.utils.audio_io import load_sequence
import torch.nn.functional as F


def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute RMS energy of a 1-D tensor."""
    return torch.sqrt(torch.mean(x ** 2) + eps)


def _mix_at_snr(primary: torch.Tensor, auxiliary: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Mix two signals at a specified SNR.

    Args:
        primary (torch.Tensor): Clean or reference sequence.
        auxiliary (torch.Tensor): Interfering or secondary sequence.
        snr_db (float): Desired signal-to-noise ratio in dB.
    """
    rp = _rms(primary)
    ra = _rms(auxiliary)
    scale = rp / (ra * (10.0 ** (snr_db / 20.0)))
    return primary + auxiliary * scale


def _mono(x: torch.Tensor) -> torch.Tensor:
    """Convert (C, T) or (T,) tensor to mono (T,)."""
    return x.mean(dim=0) if x.dim() == 2 else x.squeeze()


class SequenceDataset(Dataset):
    """
    Generic sequence-to-sequence dataset.

    Designed for tasks such as denoising, enhancement, or sequence prediction.
    Each item provides paired sequences (source and mixed) along with their
    short-time transforms.

    Returns:
        dict with:
            src_seq      : torch.Tensor (T,)
            mix_seq      : torch.Tensor (T,)
            src_stft     : torch.Tensor (F, T_frames)
            mix_stft     : torch.Tensor (F, T_frames)
            src_meta     : (path, offset)
            aux_meta     : (path, offset)
            snr_db       : float
    """

    def __init__(
        self,
        src_root: str | Path,
        aux_root: str | Path,
        partition: str,                           # 'train' | 'dev' | 'test'
        duration_sec: float,
        snr_list: List[float],
        metadata: Dict[str, Dict] | None = None,  # optional external metadata
        target_sr: int = 16_000,
        n_fft: int = 256,
        hop_length: int = 64,
        stft_window: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert partition in {"train", "dev", "test"}

        self.src_root = Path(src_root)
        self.aux_root = Path(aux_root)
        self.partition = partition
        self.duration_sec = float(duration_sec)
        self.target_sr = int(target_sr)
        self.required_samples = int(round(self.duration_sec * self.target_sr))
        self.snr_list = list(snr_list)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft_window = stft_window
        self.rng = random.Random(42)

        # Use provided metadata (e.g., pre-indexed file info)
        if metadata is None or partition not in metadata:
            raise ValueError("Metadata for partition not provided.")
        self.metadata = metadata
        self.speakers = list(self.metadata[self.partition].keys())
        print(f"Found {len(self.speakers)} source identities for {self.partition}.")

        # Index auxiliary (noise) recordings
        self.aux_entries: List[Tuple[Path, int, int]] = []
        par_dir = self.aux_root / self.partition
        for noise_type_dir in sorted([p for p in par_dir.iterdir() if p.is_dir()]):
            for wav in sorted(noise_type_dir.rglob("*.wav")):
                info = sf.info(str(wav))
                self.aux_entries.append((wav, int(info.samplerate), int(info.frames)))

        if not self.aux_entries:
            raise RuntimeError(f"No auxiliary files found under {self.aux_root}")
        print(f"Found {len(self.aux_entries)} auxiliary files for {self.partition}.")

    def __len__(self) -> int:
        return len(self.speakers)

    def _load_random_crop(self, path: Path, src_sr: int, num_frames: int) -> Tuple[torch.Tensor, int]:
        """Load a random crop from a sequence file and resample to target_sr."""
        need_src_frames = int(round(self.duration_sec * src_sr))
        max_start = max(0, num_frames - need_src_frames)
        start_src = self.rng.randint(0, max_start) if max_start > 0 else 0

        seq, src_sr = load_sequence(str(path), start_src, need_src_frames, src_sr, self.target_sr)
        T = self.required_samples
        if seq.numel() < T:
            seq = F.pad(seq, (0, T - seq.numel()))
        elif seq.numel() > T:
            seq = seq[:T]

        offset_target = int(round(start_src * (self.target_sr / float(src_sr))))
        return seq.contiguous(), offset_target

    def _random_src_entry(self, spk_id) -> Tuple[Path, int, int]:
        """Pick a random file long enough for this source ID."""
        while True:
            path, sr, n_frames = self.rng.choice(self.metadata[self.partition][spk_id])
            if n_frames / max(sr, 1) >= self.duration_sec:
                return path, sr, n_frames

    def _random_aux_entry(self) -> Tuple[Path, int, int]:
        return self.rng.choice(self.aux_entries)

    def _stft(self, seq: torch.Tensor) -> torch.Tensor:
        """Return complex STFT tensor (F, T_frames)."""
        return torch.stft(
            seq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window,
            return_complex=True,
            center=False,
            onesided=True,
        )

    def __getitem__(self, idx: int) -> Dict:
        spk_id = self.speakers[idx]
        src_path, src_sr, src_frames = self._random_src_entry(spk_id)
        src_seq, src_off = self._load_random_crop(src_path, src_sr, src_frames)

        aux_path, aux_sr, aux_frames = self._random_aux_entry()
        aux_seq, aux_off = self._load_random_crop(aux_path, aux_sr, aux_frames)

        snr_db = float(self.rng.choice(self.snr_list))
        mix_seq = _mix_at_snr(src_seq, aux_seq, snr_db)

        peak = torch.max(torch.abs(mix_seq))
        if peak > 0.999:
            mix_seq = mix_seq / (peak + 1e-8) * 0.999

        return {
            "src_seq": src_seq.to(torch.float32),
            "mix_seq": mix_seq.to(torch.float32),
            "src_stft": self._stft(src_seq).to(torch.complex64),
            "mix_stft": self._stft(mix_seq).to(torch.complex64),
            "src_meta": (str(src_path), src_off),
            "aux_meta": (str(aux_path), aux_off),
            "snr_db": snr_db,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor | List[Tuple[str, int]]]:
        src_seq = torch.stack([b["src_seq"] for b in batch], dim=0)
        mix_seq = torch.stack([b["mix_seq"] for b in batch], dim=0)
        src_stft = torch.stack([b["src_stft"] for b in batch], dim=0)
        mix_stft = torch.stack([b["mix_stft"] for b in batch], dim=0)
        src_meta = [b["src_meta"] for b in batch]
        aux_meta = [b["aux_meta"] for b in batch]
        snr_db = torch.tensor([b["snr_db"] for b in batch], dtype=torch.float32)
        return dict(
            src_seq=src_seq,
            mix_seq=mix_seq,
            src_stft=src_stft,
            mix_stft=mix_stft,
            src_meta=src_meta,
            aux_meta=aux_meta,
            snr_db=snr_db,
        )


def make_loader(
    src_root: str | Path,
    aux_root: str | Path,
    partition: str,
    duration_sec: float,
    snr_list: List[float],
    metadata: Dict[str, Dict],
    target_sr: int = 16000,
    batch_size: int = 8,
    num_workers: int = 2,
    shuffle: bool = True,
    **dataset_kwargs,
) -> Tuple[SequenceDataset, DataLoader]:
    """Helper to construct dataset + dataloader pair."""
    ds = SequenceDataset(
        src_root=src_root,
        aux_root=aux_root,
        partition=partition,
        duration_sec=duration_sec,
        snr_list=snr_list,
        metadata=metadata,
        target_sr=target_sr,
        **dataset_kwargs,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=SequenceDataset.collate_fn,
        drop_last=False,
    )
    return ds, dl
