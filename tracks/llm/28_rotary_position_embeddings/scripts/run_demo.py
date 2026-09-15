"""Show RoPE geometry and cached-attention offset equivalence."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    RoPECachedSelfAttention,
    apply_rotary_position_embeddings,
    build_rotary_cache,
)


def main() -> None:
    torch.manual_seed(28)
    rotary_cache = build_rotary_cache(max_sequence_length=8, head_dim=4)
    query = torch.randn(1, 2, 3, 4)
    rotated_query, _ = apply_rotary_position_embeddings(
        query,
        query.clone(),
        rotary_cache,
        position_offset=3,
    )
    maximum_norm_change = (
        (
            torch.linalg.vector_norm(rotated_query, dim=-1)
            - torch.linalg.vector_norm(query, dim=-1)
        )
        .abs()
        .max()
    )

    attention = RoPECachedSelfAttention(
        embed_dim=8,
        num_heads=2,
        max_sequence_length=8,
    )
    hidden_states = torch.randn(1, 5, 8)
    full_output, _ = attention(hidden_states)
    cache = None
    cached_outputs = []
    for position in range(hidden_states.shape[1]):
        output, cache = attention(
            hidden_states[:, position : position + 1],
            cache,
        )
        cached_outputs.append(output)
    cached_output = torch.cat(cached_outputs, dim=1)
    maximum_output_difference = (cached_output - full_output).abs().max()

    assert cache is not None
    print(f"rotary cache shape:       {tuple(rotary_cache.cosine.shape)}")
    print("rotation position offset: 3")
    print(f"maximum norm change:      {maximum_norm_change.item():.8f}")
    print(f"rotated KV cache shape:   {tuple(cache[0].shape)}")
    print(f"full/cached max difference:{maximum_output_difference.item():.8f}")
    print("Cached tokens use absolute offsets, so both attention paths agree.")


if __name__ == "__main__":
    main()
