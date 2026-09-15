"""Compare grouped-query and standard multi-head KV-cache sizes."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import GroupedQueryAttention, kv_cache_numel  # noqa: E402


def main() -> None:
    torch.manual_seed(29)
    batch_size = 2
    sequence_length = 6
    embed_dim = 32
    num_query_heads = 8
    num_kv_heads = 2
    head_dim = embed_dim // num_query_heads
    attention = GroupedQueryAttention(
        embed_dim,
        num_query_heads,
        num_kv_heads,
        max_sequence_length=sequence_length,
    )
    hidden_states = torch.randn(batch_size, sequence_length, embed_dim)

    full_output, full_cache = attention(hidden_states)
    cache = None
    cached_outputs = []
    for position in range(sequence_length):
        output, cache = attention(
            hidden_states[:, position : position + 1],
            cache,
        )
        cached_outputs.append(output)
    cached_output = torch.cat(cached_outputs, dim=1)

    multi_head_cache = kv_cache_numel(
        batch_size,
        sequence_length,
        num_query_heads,
        head_dim,
    )
    grouped_cache = kv_cache_numel(
        batch_size,
        sequence_length,
        num_kv_heads,
        head_dim,
    )
    maximum_difference = (cached_output - full_output).abs().max()

    assert cache is not None
    print(f"query heads:                {num_query_heads}")
    print(f"shared key/value heads:     {num_kv_heads}")
    print(f"queries per key/value head: {num_query_heads // num_kv_heads}")
    print(f"multi-head cache elements:  {multi_head_cache}")
    print(f"grouped cache elements:     {grouped_cache}")
    print(f"cache reduction:            {multi_head_cache // grouped_cache}x")
    print(f"compact cache shape:        {tuple(full_cache[0].shape)}")
    print(f"full/cached max difference: {maximum_difference.item():.8f}")
    print("KV heads expand for attention, while the persistent cache stays compact.")


if __name__ == "__main__":
    main()
