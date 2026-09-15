# Concept: many query heads, fewer key-value heads

Standard multi-head attention gives each query head a separate key head and
value head:

```text
query heads = key heads = value heads
```

Grouped-query attention (GQA) keeps the full query-head count but lets several
query heads share one key/value pair:

```text
query heads = 8
KV heads    = 2
group size  = 8 / 2 = 4 query heads per KV head
```

Query heads still produce different attention scores because their query
vectors differ. They simply compare against the same keys and aggregate the same
values within a group.

## Head mapping

With four query heads and two KV heads, consecutive grouping is:

```text
query head 0 -> KV head 0
query head 1 -> KV head 0
query head 2 -> KV head 1
query head 3 -> KV head 1
```

The implementation can express this by repeating each KV head with
`repeat_interleave`. This creates a query-head-layout view for the attention
calculation. The persistent cache should retain only the compact KV heads.

## Special cases

GQA connects two familiar attention layouts:

- `num_kv_heads == num_query_heads`: ordinary multi-head attention; every group
  has one query head.
- `num_kv_heads == 1`: multi-query attention; all query heads share one key and
  one value head.

The divisibility requirement ensures every KV head serves the same number of
query heads.

## Why the cache gets smaller

Autoregressive decoding stores keys and values but not past queries. The scalar
count for one layer is:

```text
2 * batch * sequence_length * num_kv_heads * head_dim
```

The factor two accounts for both key and value. Reducing eight KV heads to two
cuts this cache by four while preserving eight query heads.

GQA also reduces key/value projection widths and computation. This lesson
measures cache elements directly, without introducing dtype-specific byte
accounting or production memory allocators.

## RoPE and cached decoding

RoPE still rotates every new query and key at its absolute position. Query and
key head counts may differ, but `head_dim` and rotary frequencies remain shared.
New compact keys are rotated before entering the cache. Only when attention is
computed are cached keys and values repeated across query groups.
