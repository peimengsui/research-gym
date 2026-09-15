import math

import pytest
import torch
import torch.nn.functional as F

from implementation import (
    GatedLinearAttention,
    gate_log_decay,
    gated_linear_attention_parallel,
    gated_linear_attention_recurrent,
)
from provided import GatedLinearAttentionState


def test_gate_log_decay_matches_stable_parameterization() -> None:
    logits = torch.tensor([-4.0, 0.0, 3.0])

    log_decay = gate_log_decay(logits, normalizer=8.0)
    decay = log_decay.exp()

    assert torch.equal(log_decay, F.logsigmoid(logits) / 8.0)
    assert ((decay > 0.0) & (decay < 1.0)).all()


def test_larger_normalizer_retains_memory_longer() -> None:
    logits = torch.zeros(4)

    short_decay = gate_log_decay(logits, normalizer=1.0).exp()
    long_decay = gate_log_decay(logits, normalizer=16.0).exp()

    assert (long_decay > short_decay).all()


def test_recurrent_update_selectively_forgets_key_channels() -> None:
    query = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    value = torch.tensor([[[[2.0], [0.0]]]])
    log_decay = torch.tensor([[[[0.0, 0.0], [math.log(0.25), 0.0]]]])

    output, state = gated_linear_attention_recurrent(
        query,
        key,
        value,
        log_decay,
        scale=1.0,
    )

    assert torch.allclose(output.flatten(), torch.tensor([2.0, 0.5]))
    assert torch.allclose(state.memory[0, 0, :, 0], torch.tensor([0.5, 0.0]))


def test_parallel_and_recurrent_gla_match() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 3, 7, 4)
    key = torch.randn(2, 3, 7, 4)
    value = torch.randn(2, 3, 7, 5)
    log_decay = gate_log_decay(torch.randn_like(query), normalizer=4.0)

    parallel, parallel_state = gated_linear_attention_parallel(
        query,
        key,
        value,
        log_decay,
    )
    recurrent, recurrent_state = gated_linear_attention_recurrent(
        query,
        key,
        value,
        log_decay,
    )

    assert torch.allclose(parallel, recurrent, atol=1e-5)
    assert torch.allclose(parallel_state.memory, recurrent_state.memory, atol=1e-5)


def test_both_forms_continue_from_incoming_state() -> None:
    torch.manual_seed(1)
    query = torch.randn(1, 2, 5, 3)
    key = torch.randn(1, 2, 5, 3)
    value = torch.randn(1, 2, 5, 4)
    log_decay = gate_log_decay(torch.randn_like(query), normalizer=3.0)
    initial = GatedLinearAttentionState(torch.randn(1, 2, 3, 4))
    saved_initial = initial.memory.clone()

    parallel, parallel_state = gated_linear_attention_parallel(
        query,
        key,
        value,
        log_decay,
        initial,
    )
    recurrent, recurrent_state = gated_linear_attention_recurrent(
        query,
        key,
        value,
        log_decay,
        initial,
    )

    assert torch.allclose(parallel, recurrent, atol=1e-5)
    assert torch.allclose(parallel_state.memory, recurrent_state.memory, atol=1e-5)
    assert torch.equal(initial.memory, saved_initial)


def test_gla_is_causal() -> None:
    torch.manual_seed(2)
    query = torch.randn(1, 1, 6, 4)
    key = torch.randn(1, 1, 6, 4)
    value = torch.randn(1, 1, 6, 3)
    log_decay = gate_log_decay(torch.randn_like(query))
    changed_key = key.clone()
    changed_value = value.clone()
    changed_decay = log_decay.clone()
    changed_key[:, :, 4:] += 100.0
    changed_value[:, :, 4:] -= 100.0
    changed_decay[:, :, 4:] -= 5.0

    original, _ = gated_linear_attention_parallel(query, key, value, log_decay)
    changed, _ = gated_linear_attention_parallel(
        query,
        changed_key,
        changed_value,
        changed_decay,
    )

    assert torch.allclose(original[:, :, :4], changed[:, :, :4])


def test_module_parallel_and_streaming_recurrent_paths_match() -> None:
    torch.manual_seed(3)
    attention = GatedLinearAttention(embed_dim=16, num_heads=4, gate_rank=3)
    x = torch.randn(2, 9, 16)

    full_output, full_state = attention(x)
    prefix_output, state = attention(x[:, :4], use_recurrent=True)
    suffix_output, state = attention(x[:, 4:], state, use_recurrent=True)

    assert full_output.shape == x.shape
    assert torch.allclose(
        torch.cat((prefix_output, suffix_output), dim=1),
        full_output,
        atol=1e-5,
    )
    assert state.memory.shape == full_state.memory.shape == (2, 4, 4, 4)
    assert torch.allclose(state.memory, full_state.memory, atol=1e-5)
    assert attention.gate_down.out_features == 3
    assert attention.gate_up.in_features == 3


def test_state_size_does_not_grow_with_sequence_length() -> None:
    torch.manual_seed(4)
    attention = GatedLinearAttention(embed_dim=12, num_heads=3)

    _, short_state = attention(torch.randn(1, 2, 12), use_recurrent=True)
    _, long_state = attention(torch.randn(1, 20, 12), use_recurrent=True)

    assert short_state.memory.shape == long_state.memory.shape == (1, 3, 4, 4)
    assert short_state.memory.numel() == long_state.memory.numel() == 48


def test_gradients_reach_attention_forget_and_output_gates() -> None:
    torch.manual_seed(5)
    attention = GatedLinearAttention(embed_dim=12, num_heads=3, gate_rank=2)
    x = torch.randn(2, 6, 12, requires_grad=True)

    output, _ = attention(x)
    output.square().mean().backward()

    assert x.grad is not None
    for projection in (
        attention.query,
        attention.key,
        attention.value,
        attention.gate_down,
        attention.gate_up,
        attention.output_gate,
        attention.output,
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()


def test_validation_rejects_growing_gate_and_bad_state() -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 5)
    growing_log_decay = torch.full_like(query, 0.1)
    bad_state = GatedLinearAttentionState(torch.zeros(1, 2, 4, 4))

    with pytest.raises(ValueError, match="non-positive"):
        gated_linear_attention_recurrent(
            query,
            key,
            value,
            growing_log_decay,
        )
    with pytest.raises(ValueError, match="incompatible shape"):
        gated_linear_attention_recurrent(
            query,
            key,
            value,
            torch.zeros_like(query),
            bad_state,
        )
