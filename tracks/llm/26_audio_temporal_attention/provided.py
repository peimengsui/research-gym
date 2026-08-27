"""Completed spectrogram embedding and Transformer utilities from prior lessons."""

import torch
from torch import nn


def waveform_to_log_spectrogram(
    waveforms: torch.Tensor, n_fft: int, hop_length: int
) -> torch.Tensor:
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
    batch, frequency_bins, time_frames = spectrograms.shape
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


class AudioSpectrogramEmbedding(nn.Module):
    """Completed fixed-size audio embedder carried forward from llm.25."""

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
        self.frequency_bin_count = n_fft // 2 + 1
        self.frame_count = 1 + (sample_count - n_fft) // hop_length
        if self.frequency_bin_count % frequency_patch_size != 0:
            raise ValueError("frequency bins must divide into patches")
        if self.frame_count % temporal_patch_size != 0:
            raise ValueError("STFT frames must divide into patches")
        self.frequency_token_count = self.frequency_bin_count // frequency_patch_size
        self.temporal_token_count = self.frame_count // temporal_patch_size
        self.num_tokens = self.frequency_token_count * self.temporal_token_count
        patch_dim = frequency_patch_size * temporal_patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.temporal_position_embedding = nn.Embedding(
            self.temporal_token_count, embed_dim
        )
        self.frequency_position_embedding = nn.Embedding(
            self.frequency_token_count, embed_dim
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        expected = (waveforms.shape[0], self.sample_count)
        if waveforms.ndim != 2 or waveforms.shape != expected:
            raise ValueError(f"waveforms must have shape (batch, {self.sample_count})")
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


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, tokens, embed_dim = x.shape
    return x.reshape(batch, tokens, num_heads, embed_dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
