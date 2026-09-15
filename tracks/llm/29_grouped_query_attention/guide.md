# Implementation guide

Open `implementation.py` in your lesson workspace and complete the four TODOs.
RoPE, causal masking, scaled attention, projection setup, and validation are
already provided.

## 1. Expand compact KV heads

The provided validator returns:

```text
queries_per_kv_head = num_query_heads // num_kv_heads
```

Use `repeat_interleave` along head dimension one. Repeating individual heads is
important: for two KV heads and group size three, the mapping should be
`[0, 0, 0, 1, 1, 1]`, not `[0, 1, 0, 1, 0, 1]`.

## 2. Compute grouped-query attention

Inputs use:

```text
query:     [batch, query_heads, query_time, head_dim]
key/value: [batch, kv_heads, key_time, head_dim]
```

Expand key and value to `query.shape[1]` heads, then call the provided scaled
dot-product attention function. The output retains query-head layout.

When `kv_heads == query_heads`, each head repeats once and this operation is
exactly standard multi-head attention.

## 3. Count cache elements

Multiply batch size, sequence length, KV-head count, and head dimension. Multiply
again by two for key and value. The query-head count is intentionally absent:
past queries are not stored for decoding.

## 4. Integrate GQA with RoPE and caching

The query projection has width `embed_dim`. Key and value projections have the
smaller width:

```text
num_kv_heads * head_dim
```

Reshape query with `num_query_heads` and key/value with `num_kv_heads`. Apply the
provided RoPE function to new queries and keys using `past_length` as the
absolute position offset.

Append the compact rotated keys and raw values along time dimension two. Pass
that compact cache to `grouped_query_attention`, which handles temporary
expansion. Merge query heads and apply the output projection.

## Common bugs

- using `repeat` and alternating KV heads instead of forming consecutive groups
- requiring query, key, and value to have the same head count before expansion
- projecting keys and values to `embed_dim`, losing parameter and cache savings
- storing repeated query-head-layout keys and values in the persistent cache
- using query-head count in the cache-size formula
- applying RoPE after KV expansion when compact keys could be rotated once
- using zero as the RoPE position offset during cached decoding
- concatenating compact cache tensors along the KV-head dimension instead of time
