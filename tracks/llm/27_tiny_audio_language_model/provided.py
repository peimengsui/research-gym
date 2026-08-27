"""Completed variable-duration audio encoder and multimodal Transformer pieces."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, tokens, embed_dim = x.shape
    return x.reshape(batch, tokens, num_heads, embed_dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AudioSpectrogramEmbedding(nn.Module):
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
        self.sample_count = sample_count
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frequency_patch_size = frequency_patch_size
        self.temporal_patch_size = temporal_patch_size
        frequency_bins = n_fft // 2 + 1
        frame_count = 1 + (sample_count - n_fft) // hop_length
        if sample_count < n_fft or frequency_bins % frequency_patch_size != 0:
            raise ValueError("audio dimensions must divide into patches")
        if frame_count % temporal_patch_size != 0:
            raise ValueError("STFT frames must divide into temporal patches")
        self.frequency_token_count = frequency_bins // frequency_patch_size
        self.temporal_token_count = frame_count // temporal_patch_size
        self.num_tokens = self.frequency_token_count * self.temporal_token_count
        self.projection = nn.Linear(
            frequency_patch_size * temporal_patch_size, embed_dim
        )
        self.temporal_position_embedding = nn.Embedding(
            self.temporal_token_count, embed_dim
        )
        self.frequency_position_embedding = nn.Embedding(
            self.frequency_token_count, embed_dim
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2 or waveforms.shape[1] != self.sample_count:
            raise ValueError(f"waveforms must have shape (batch, {self.sample_count})")
        window = torch.hann_window(
            self.n_fft, device=waveforms.device, dtype=waveforms.dtype
        )
        spectrum = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=False,
            return_complex=True,
        )
        spectrograms = torch.log1p(spectrum.abs())
        batch = waveforms.shape[0]
        patches = spectrograms.unfold(
            1, self.frequency_patch_size, self.frequency_patch_size
        ).unfold(2, self.temporal_patch_size, self.temporal_patch_size)
        patches = (
            patches.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(batch, self.num_tokens, -1)
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


def _audio_token_validity(
    sample_lengths: torch.Tensor,
    embedding: AudioSpectrogramEmbedding,
) -> torch.Tensor:
    counts = 1 + torch.div(
        sample_lengths - embedding.n_fft,
        embedding.hop_length,
        rounding_mode="floor",
    )
    counts = torch.where(
        sample_lengths >= embedding.n_fft, counts, torch.zeros_like(counts)
    )
    temporal_counts = torch.div(
        counts, embedding.temporal_patch_size, rounding_mode="floor"
    )
    temporal_ids = torch.arange(
        embedding.temporal_token_count, device=sample_lengths.device
    )
    temporal_validity = temporal_ids.unsqueeze(0) < temporal_counts.unsqueeze(1)
    return temporal_validity.repeat_interleave(embedding.frequency_token_count, dim=1)


def _safe_attention_weights(
    scores: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    expanded_mask = attention_mask.unsqueeze(1)
    weights = F.softmax(scores.masked_fill(~expanded_mask, float("-inf")), dim=-1)
    return torch.nan_to_num(weights, nan=0.0).masked_fill(~expanded_mask, 0.0)


class AudioSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must positively divide embed_dim")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        mask = validity.unsqueeze(2) & validity.unsqueeze(1)
        weights = _safe_attention_weights(scores, mask)
        output = self.output(merge_heads(weights @ value))
        return output.masked_fill(~validity.unsqueeze(-1), 0.0)


class AudioTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = AudioSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), validity)
        x = x + self.feed_forward(self.norm2(x))
        return x.masked_fill(~validity.unsqueeze(-1), 0.0)


class TinyVariableAudioEncoder(nn.Module):
    """Completed variable-duration encoder carried forward from llm.26."""

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
        self.embedding = AudioSpectrogramEmbedding(
            sample_count,
            n_fft,
            hop_length,
            frequency_patch_size,
            temporal_patch_size,
            embed_dim,
        )
        self.num_tokens = self.embedding.num_tokens
        self.blocks = nn.ModuleList(
            [AudioTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, waveforms: torch.Tensor, sample_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample_lengths.shape != (waveforms.shape[0],):
            raise ValueError("sample_lengths must match waveform batch")
        if (sample_lengths < 0).any() or (
            sample_lengths > self.embedding.sample_count
        ).any():
            raise ValueError("sample_lengths must fit padded waveforms")
        validity = _audio_token_validity(sample_lengths, self.embedding)
        x = self.embedding(waveforms).masked_fill(~validity.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, validity)
        x = self.final_norm(x).masked_fill(~validity.unsqueeze(-1), 0.0)
        return x, validity


class MultimodalSelfAttention(nn.Module):
    """Completed safe masked attention for the unified audio-text sequence."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        weights = _safe_attention_weights(scores, attention_mask)
        output = self.output(merge_heads(weights @ value))
        query_validity = attention_mask.any(dim=-1)
        return output.masked_fill(~query_validity.unsqueeze(-1), 0.0)


class MultimodalTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = MultimodalSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query_validity = attention_mask.any(dim=-1)
        x = x + self.attention(self.norm1(x), attention_mask)
        x = x + self.feed_forward(self.norm2(x))
        return x.masked_fill(~query_validity.unsqueeze(-1), 0.0)


@dataclass(frozen=True)
class TinyAudioVocabulary:
    tokens: tuple[str, ...] = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<user>",
        "<assistant>",
        "what",
        "tone",
        "low",
        "high",
        "audio",
    )

    @property
    def eos_token_id(self) -> int:
        return self.tokens.index("<eos>")

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.tokens.index(token) for token in tokens]

    def decode(self, token_ids: torch.Tensor) -> list[list[str]]:
        return [[self.tokens[index] for index in row] for row in token_ids.tolist()]


def make_toy_tone_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded low/high sine waves and their original sample lengths."""

    sample_index = torch.arange(18, dtype=torch.float32)
    low = torch.sin(2 * torch.pi * sample_index / 6)
    high = torch.sin(2 * torch.pi * 2 * sample_index / 6)
    high[14:] = 0.0
    return torch.stack((low, high)), torch.tensor([18, 14])
