"""Show learned GLA decay, recurrent equivalence, and fixed-size memory."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    gate_log_decay,
    gated_linear_attention_parallel,
    gated_linear_attention_recurrent,
)


def main() -> None:
    torch.manual_seed(30)
    batch_size = 1
    num_heads = 4
    sequence_length = 32
    key_dim = 8
    value_dim = 8
    query = torch.randn(batch_size, num_heads, sequence_length, key_dim)
    key = torch.randn_like(query)
    value = torch.randn(batch_size, num_heads, sequence_length, value_dim)
    gate_logits = torch.randn_like(query)
    log_decay = gate_log_decay(gate_logits)

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

    maximum_difference = (parallel - recurrent).abs().max().item()
    state_difference = (
        (parallel_state.memory - recurrent_state.memory).abs().max().item()
    )
    decay = log_decay.exp()
    state_elements = recurrent_state.memory.numel()
    kv_cache_elements = batch_size * num_heads * sequence_length * (key_dim + value_dim)

    print(f"learned decay range:              {decay.min():.4f} to {decay.max():.4f}")
    print(f"parallel/recurrent max diff:     {maximum_difference:.8f}")
    print(f"final-state max diff:            {state_difference:.8f}")
    print(f"softmax-style KV-cache elements: {kv_cache_elements}")
    print(f"fixed GLA state elements:        {state_elements}")
    print(f"memory ratio at this length:     {kv_cache_elements / state_elements:.2f}x")
    print("GLA learns what to forget while retaining fixed-size matrix memory.")


if __name__ == "__main__":
    main()
