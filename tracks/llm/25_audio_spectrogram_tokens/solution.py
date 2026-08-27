"""Reference solution for waveform spectrogram patch embeddings."""

import torch
from torch import nn


def waveform_to_log_spectrogram(
    waveforms: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Convert `(batch, samples)` waveforms to log-magnitude spectrograms."""

    if waveforms.ndim != 2 or not waveforms.is_floating_point():
        raise ValueError("waveforms must be a 2D floating-point tensor")
    if n_fft < 2 or hop_length <= 0:
        raise ValueError("n_fft must be at least 2 and hop_length must be positive")
    if waveforms.shape[0] == 0 or waveforms.shape[1] < n_fft:
        raise ValueError("waveforms must contain a batch and at least n_fft samples")

    window = torch.hann_window(n_fft, device=waveforms.device, dtype=waveforms.dtype)
    spectrum = torch.stft(
        waveforms,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )
    return torch.log1p(spectrum.abs())


def spectrogram_patchify(
    spectrograms: torch.Tensor,
    frequency_patch_size: int,
    temporal_patch_size: int,
) -> torch.Tensor:
    """Return time-major patches from `(B, frequency, time)` spectrograms."""

    if spectrograms.ndim != 3:
        raise ValueError("spectrograms must have shape (batch, frequency, time)")
    if frequency_patch_size <= 0 or temporal_patch_size <= 0:
        raise ValueError("patch sizes must be positive")
    batch, frequency_bins, time_frames = spectrograms.shape
    if frequency_bins % frequency_patch_size != 0:
        raise ValueError("frequency bins must be divisible by frequency_patch_size")
    if time_frames % temporal_patch_size != 0:
        raise ValueError("time frames must be divisible by temporal_patch_size")

    frequency_grid = frequency_bins // frequency_patch_size
    temporal_grid = time_frames // temporal_patch_size
    patches = spectrograms.unfold(1, frequency_patch_size, frequency_patch_size).unfold(
        2, temporal_patch_size, temporal_patch_size
    )
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    return patches.reshape(
        batch,
        temporal_grid * frequency_grid,
        frequency_patch_size * temporal_patch_size,
    )


def spectrogram_unpatchify(
    patches: torch.Tensor,
    frequency_bins: int,
    time_frames: int,
    frequency_patch_size: int,
    temporal_patch_size: int,
) -> torch.Tensor:
    """Reconstruct `(B, frequency, time)` spectrograms from time-major patches."""

    if patches.ndim != 3:
        raise ValueError("patches must have shape (batch, patch_count, patch_dim)")
    if (
        min(
            frequency_bins,
            time_frames,
            frequency_patch_size,
            temporal_patch_size,
        )
        <= 0
    ):
        raise ValueError("spectrogram and patch dimensions must be positive")
    if frequency_bins % frequency_patch_size != 0:
        raise ValueError("frequency bins must be divisible by frequency_patch_size")
    if time_frames % temporal_patch_size != 0:
        raise ValueError("time frames must be divisible by temporal_patch_size")

    batch, patch_count, patch_dim = patches.shape
    frequency_grid = frequency_bins // frequency_patch_size
    temporal_grid = time_frames // temporal_patch_size
    expected_count = temporal_grid * frequency_grid
    expected_dim = frequency_patch_size * temporal_patch_size
    if (patch_count, patch_dim) != (expected_count, expected_dim):
        raise ValueError("patch shape does not match the requested spectrogram")
    grid = patches.reshape(
        batch,
        temporal_grid,
        frequency_grid,
        frequency_patch_size,
        temporal_patch_size,
    )
    grid = grid.permute(0, 2, 3, 1, 4).contiguous()
    return grid.reshape(batch, frequency_bins, time_frames)


class AudioSpectrogramEmbedding(nn.Module):
    """Project fixed-size spectrogram patches and add factorized positions."""

    def __init__(
        self,
        sample_count: int,
        n_fft: int,
        hop_length: int,
        frequency_patch_size: int,
        temporal_patch_size: int,
        embed_dim: int,
    ):
        super().__init__()
        if (
            min(
                sample_count,
                n_fft,
                hop_length,
                frequency_patch_size,
                temporal_patch_size,
                embed_dim,
            )
            <= 0
        ):
            raise ValueError("all dimensions must be positive")
        if sample_count < n_fft:
            raise ValueError("sample_count must be at least n_fft")
        self.sample_count = sample_count
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frequency_patch_size = frequency_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        self.frequency_bin_count = n_fft // 2 + 1
        self.frame_count = 1 + (sample_count - n_fft) // hop_length
        if self.frequency_bin_count % frequency_patch_size != 0:
            raise ValueError("frequency bins must be divisible by frequency_patch_size")
        if self.frame_count % temporal_patch_size != 0:
            raise ValueError("STFT frames must be divisible by temporal_patch_size")
        self.frequency_token_count = self.frequency_bin_count // frequency_patch_size
        self.temporal_token_count = self.frame_count // temporal_patch_size
        self.num_tokens = self.frequency_token_count * self.temporal_token_count
        self.patch_dim = frequency_patch_size * temporal_patch_size

        self.projection = nn.Linear(self.patch_dim, embed_dim)
        self.temporal_position_embedding = nn.Embedding(
            self.temporal_token_count, embed_dim
        )
        self.frequency_position_embedding = nn.Embedding(
            self.frequency_token_count, embed_dim
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return audio embeddings shaped `(batch, num_tokens, embed_dim)`."""

        if (
            waveforms.ndim != 2
            or waveforms.shape[0] == 0
            or waveforms.shape[1] != self.sample_count
            or not waveforms.is_floating_point()
        ):
            raise ValueError(
                f"waveforms must be floating point with shape (batch, {self.sample_count})"
            )
        spectrograms = waveform_to_log_spectrogram(
            waveforms, self.n_fft, self.hop_length
        )
        patches = spectrogram_patchify(
            spectrograms,
            self.frequency_patch_size,
            self.temporal_patch_size,
        )
        temporal_ids = torch.arange(
            self.temporal_token_count, device=waveforms.device
        ).repeat_interleave(self.frequency_token_count)
        frequency_ids = torch.arange(
            self.frequency_token_count, device=waveforms.device
        ).repeat(self.temporal_token_count)
        return (
            self.projection(patches)
            + self.temporal_position_embedding(temporal_ids)
            + self.frequency_position_embedding(frequency_ids)
        )
