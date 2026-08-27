"""Learner scaffold for turning waveforms into spectrogram patch tokens."""

import torch
from torch import nn


def waveform_to_log_spectrogram(
    waveforms: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Convert `(batch, samples)` waveforms to `(batch, frequency, time)`.

    The output contains `log(1 + magnitude)` values from a non-centered STFT.
    """

    if waveforms.ndim != 2 or not waveforms.is_floating_point():
        raise ValueError("waveforms must be a 2D floating-point tensor")
    if n_fft < 2 or hop_length <= 0:
        raise ValueError("n_fft must be at least 2 and hop_length must be positive")
    if waveforms.shape[0] == 0 or waveforms.shape[1] < n_fft:
        raise ValueError("waveforms must contain a batch and at least n_fft samples")

    # TODO 1: Create a Hann window on the waveform device/dtype, compute a
    # non-centered complex STFT, and return log1p of its magnitude.
    raise NotImplementedError


def spectrogram_patchify(
    spectrograms: torch.Tensor,
    frequency_patch_size: int,
    temporal_patch_size: int,
) -> torch.Tensor:
    """Return time-major patches from `(B, frequency, time)` spectrograms.

    Output shape: `(B, patch_count, frequency_patch_size * temporal_patch_size)`.
    """

    if spectrograms.ndim != 3:
        raise ValueError("spectrograms must have shape (batch, frequency, time)")
    if frequency_patch_size <= 0 or temporal_patch_size <= 0:
        raise ValueError("patch sizes must be positive")
    batch, frequency_bins, time_frames = spectrograms.shape
    if frequency_bins % frequency_patch_size != 0:
        raise ValueError("frequency bins must be divisible by frequency_patch_size")
    if time_frames % temporal_patch_size != 0:
        raise ValueError("time frames must be divisible by temporal_patch_size")

    # TODO 2: Unfold frequency and time, arrange the grid in time-major order,
    # and flatten grid/content dimensions into tokens and patch features.
    raise NotImplementedError


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

    # TODO 3: Validate token/feature dimensions, restore the time-major patch
    # grid, permute neighboring grid/patch axes together, and reshape.
    raise NotImplementedError


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

        # TODO 4: Create a patch projection and factorized temporal/frequency
        # position embedding tables with explicit attribute names.
        raise NotImplementedError

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

        # TODO 5: Compute and patchify the spectrogram, project content, and add
        # temporal/frequency positions in matching time-major order.
        raise NotImplementedError
