"""Provided state containers, shape utilities, normalization, and validation."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GatedLinearAttentionState:
    """Fixed-size matrix memory carried between sequence chunks."""

    memory: torch.Tensor


def validate_gate_logits(gate_logits: torch.Tensor, normalizer: float) -> None:
    """Validate unconstrained logits before converting them to log decays."""

    if gate_logits.numel() == 0 or not torch.is_floating_point(gate_logits):
        raise ValueError("gate_logits must be a non-empty floating-point tensor")
    if not torch.isfinite(gate_logits).all():
        raise ValueError("gate_logits must contain finite values")
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("normalizer must be finite and positive")


def resolve_scale(scale: float | None, key_dim: int) -> float:
    """Return an explicit positive query-key scale."""

    resolved = key_dim**-0.5 if scale is None else scale
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("scale must be finite and positive")
    return resolved


def validate_gla_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    scale: float | None,
) -> tuple[int, int, int, int, int, float]:
    """Validate head-layout GLA tensors and return their dimensions and scale."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must use four-dimensional head layout")
    if query.shape != key.shape:
        raise ValueError("query and key must have the same shape")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("value must share query batch, heads, and time")
    if query.shape[2] == 0 or log_decay.shape != query.shape:
        raise ValueError("time must be positive and log_decay must match query shape")
    tensors = (query, key, value, log_decay)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all GLA tensors must share a device")
    if any(tensor.dtype != query.dtype for tensor in tensors):
        raise ValueError("all GLA tensors must share a dtype")
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("all GLA tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("all GLA tensors must contain finite values")
    if (log_decay > 0.0).any():
        raise ValueError("log_decay must be non-positive so decay cannot exceed one")
    batch_size, num_heads, sequence_length, key_dim = query.shape
    resolved_scale = resolve_scale(scale, key_dim)
    return (
        batch_size,
        num_heads,
        sequence_length,
        key_dim,
        value.shape[3],
        resolved_scale,
    )


def validate_state(
    state: GatedLinearAttentionState,
    batch_size: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    reference: torch.Tensor,
) -> None:
    """Validate recurrent matrix memory against current GLA inputs."""

    expected_shape = (batch_size, num_heads, key_dim, value_dim)
    if state.memory.shape != expected_shape:
        raise ValueError("state.memory has incompatible shape")
    if state.memory.device != reference.device or state.memory.dtype != reference.dtype:
        raise ValueError("state and GLA inputs must share device and dtype")
    if (
        not torch.is_floating_point(state.memory)
        or not torch.isfinite(state.memory).all()
    ):
        raise ValueError("state.memory must contain finite floating-point values")


def validate_module_configuration(
    embed_dim: int,
    num_heads: int,
    gate_rank: int,
    gate_logit_normalizer: float,
    norm_epsilon: float,
) -> int:
    """Validate educational GLA dimensions and return per-head width."""

    if embed_dim <= 0 or num_heads <= 0 or gate_rank <= 0:
        raise ValueError("embed_dim, num_heads, and gate_rank must be positive")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    validate_gate_logits(torch.zeros(1), gate_logit_normalizer)
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0.0:
        raise ValueError("norm_epsilon must be finite and positive")
    return embed_dim // num_heads


def validate_module_forward(
    x: torch.Tensor,
    state: GatedLinearAttentionState | None,
    embed_dim: int,
    num_heads: int,
    head_dim: int,
    use_recurrent: bool,
) -> None:
    """Validate hidden states, optional memory, and execution selection."""

    if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != embed_dim:
        raise ValueError("x must have shape [batch, positive_time, embed_dim]")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("x must contain finite floating-point values")
    if not isinstance(use_recurrent, bool):
        raise ValueError("use_recurrent must be a boolean")
    if state is not None:
        validate_state(
            state,
            x.shape[0],
            num_heads,
            head_dim,
            head_dim,
            x,
        )


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape [batch, time, embed] into [batch, heads, time, head_dim]."""

    batch_size, sequence_length, embed_dim = x.shape
    if embed_dim % num_heads != 0:
        raise ValueError("projection width must be divisible by num_heads")
    head_dim = embed_dim // num_heads
    return x.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Reshape [batch, heads, time, head_dim] into [batch, time, embed]."""

    batch_size, num_heads, sequence_length, head_dim = x.shape
    return x.transpose(1, 2).reshape(
        batch_size,
        sequence_length,
        num_heads * head_dim,
    )


def rms_norm_per_head(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Apply non-affine RMS normalization across each head's value channels."""

    if x.ndim != 4 or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("x must use head layout and epsilon must be positive")
    inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon)
    return x * inverse_rms
