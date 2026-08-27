"""Reference solution for variable-duration audio attention."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from provided import AudioSpectrogramEmbedding, FeedForward, merge_heads, split_heads


def valid_stft_frame_counts(
    sample_lengths: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Return valid non-centered STFT frame counts for `(batch,)` lengths."""

    if sample_lengths.ndim != 1 or sample_lengths.dtype != torch.long:
        raise ValueError("sample_lengths must be a 1D torch.long tensor")
    if sample_lengths.numel() == 0 or (sample_lengths < 0).any():
        raise ValueError("sample_lengths must be non-empty and non-negative")
    if n_fft < 2 or hop_length <= 0:
        raise ValueError("n_fft must be at least 2 and hop_length must be positive")
    counts = 1 + torch.div(sample_lengths - n_fft, hop_length, rounding_mode="floor")
    return torch.where(sample_lengths >= n_fft, counts, torch.zeros_like(counts))


def make_audio_token_validity(
    sample_lengths: torch.Tensor,
    n_fft: int,
    hop_length: int,
    temporal_patch_size: int,
    frequency_token_count: int,
    max_temporal_tokens: int,
) -> torch.Tensor:
    """Return time-major token validity shaped `(batch, audio_tokens)`."""

    if min(temporal_patch_size, frequency_token_count, max_temporal_tokens) <= 0:
        raise ValueError("token grid dimensions must be positive")
    frame_counts = valid_stft_frame_counts(sample_lengths, n_fft, hop_length)
    valid_temporal_tokens = torch.div(
        frame_counts, temporal_patch_size, rounding_mode="floor"
    )
    if (valid_temporal_tokens > max_temporal_tokens).any():
        raise ValueError("sample lengths exceed the maximum temporal token grid")
    temporal_indices = torch.arange(max_temporal_tokens, device=sample_lengths.device)
    temporal_validity = temporal_indices.unsqueeze(0) < valid_temporal_tokens.unsqueeze(
        1
    )
    return temporal_validity.repeat_interleave(frequency_token_count, dim=1)


def make_audio_attention_mask(validity: torch.Tensor) -> torch.Tensor:
    """Return `(batch, query, key)` validity for bidirectional audio attention."""

    if validity.ndim != 2 or validity.dtype != torch.bool:
        raise ValueError("validity must be a 2D boolean tensor")
    return validity.unsqueeze(2) & validity.unsqueeze(1)


def safe_masked_softmax(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax attention scores while returning zeros for fully masked rows."""

    if scores.ndim != 4:
        raise ValueError("scores must have shape (batch, heads, query, key)")
    expected = (scores.shape[0], scores.shape[2], scores.shape[3])
    if attention_mask.shape != expected or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean (batch, query, key)")
    expanded_mask = attention_mask.unsqueeze(1)
    weights = F.softmax(scores.masked_fill(~expanded_mask, float("-inf")), dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return weights.masked_fill(~expanded_mask, 0.0)


class AudioSelfAttention(nn.Module):
    """Bidirectional self-attention over valid audio tokens."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must positively divide embed_dim")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        validity: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"x must have shape (batch, tokens, {self.embed_dim})")
        if validity.shape != x.shape[:2] or validity.dtype != torch.bool:
            raise ValueError("validity must be boolean and match x token dimensions")
        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = safe_masked_softmax(scores, make_audio_attention_mask(validity))
        output = self.output(merge_heads(weights @ value))
        output = output.masked_fill(~validity.unsqueeze(-1), 0.0)
        if return_weights:
            return output, weights
        return output


class AudioTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = AudioSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        validity: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(self.norm1(x), validity, return_weights=True)
        x = x + attended
        x = x + self.feed_forward(self.norm2(x))
        x = x.masked_fill(~validity.unsqueeze(-1), 0.0)
        if return_weights:
            return x, weights
        return x


class TinyVariableAudioEncoder(nn.Module):
    """Encode a padded waveform batch while preserving token validity."""

    def __init__(
        self,
        sample_count: int,
        n_fft: int,
        hop_length: int,
        frequency_patch_size: int,
        temporal_patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.embedding = AudioSpectrogramEmbedding(
            sample_count,
            n_fft,
            hop_length,
            frequency_patch_size,
            temporal_patch_size,
            embed_dim,
        )
        self.blocks = nn.ModuleList(
            [AudioTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
        return_weights: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        if sample_lengths.shape != (waveforms.shape[0],):
            raise ValueError("sample_lengths must match the waveform batch")
        if (sample_lengths > self.embedding.sample_count).any():
            raise ValueError("sample_lengths cannot exceed padded sample_count")
        validity = make_audio_token_validity(
            sample_lengths,
            self.embedding.n_fft,
            self.embedding.hop_length,
            self.embedding.temporal_patch_size,
            self.embedding.frequency_token_count,
            self.embedding.temporal_token_count,
        )
        x = self.embedding(waveforms)
        x = x.masked_fill(~validity.unsqueeze(-1), 0.0)
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, validity, return_weights=True)
            attention_maps.append(weights)
        x = self.final_norm(x).masked_fill(~validity.unsqueeze(-1), 0.0)
        if return_weights:
            return x, validity, attention_maps
        return x, validity
