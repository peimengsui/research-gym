# Implementation guide

Open `implementation.py` in your lesson workspace and complete the four TODOs.
Tensor/state validation, head reshaping, and RMS normalization are provided.

## 1. Parameterize the forget gate

Convert unconstrained logits to non-positive log decays:

```python
F.logsigmoid(gate_logits) / normalizer
```

Exponentiating this gives `sigmoid(logits) ** (1 / normalizer)`, which lies
between zero and one. Keep it in log space until the recurrence needs the
current decay.

## 2. Implement recurrent GLA

Initialize matrix memory with:

```text
[batch, heads, key_dim, value_dim]
```

At each position:

1. Exponentiate that position's log decay and add a trailing singleton axis.
2. Multiply the old memory by this per-key-channel decay.
3. Add the outer product of the current key and value.
4. Contract the current query with updated memory and multiply by the resolved
   scale.

Return new state instead of changing incoming state in place.

## 3. Implement the explicit parallel oracle

Use a cumulative sum of log decays. If `L_t` is the cumulative log decay at
target `t`, then a write at source `i` is retained by:

```text
exp(L_t - L_i)    for i <= t
```

The subtraction excludes the source token's gate because the recurrence gates
old memory before adding the current write. Apply a causal lower-triangular
mask before exponentiating.

Combine query, key, and the per-channel retention tensor to produce scalar
source weights, then combine those weights with values. If incoming state is
present, its retention at target `t` is `exp(L_t)` because every gate in the
new chunk acts on that old memory.

Use the last target's retention factors to construct final memory. This
explicit implementation is intentionally quadratic and meant for verification,
not performance.

## 4. Assemble the GLA layer

Project and split query, key, and value. Produce gate logits through the
low-rank path:

```text
x -> gate_down -> gate_up -> split_heads
```

Call `gate_log_decay`, then select the recurrent or parallel implementation.
After the GLA read:

1. apply the provided per-head RMS normalization
2. project `x` through `output_gate`, split heads, and apply SiLU
3. multiply normalized values by that gate
4. merge heads and apply the final output projection

The memory forget gate and output gate have different jobs. Do not reuse one
for the other.

## Common bugs

- using `sigmoid(logits / normalizer)` instead of
  `sigmoid(logits) ** (1 / normalizer)`
- applying the current decay to the current key/value write
- reading memory before adding the current write
- using a scalar gate instead of one gate per key channel
- dividing by a key-sum normalizer carried over from older kernel attention
- applying softmax to GLA query-key weights
- forgetting that incoming state is decayed by the current token's gate
- normalizing across all heads rather than independently per head
- confusing the forget gate with the separate SiLU output gate
- describing the explicit parallel oracle as the hardware-efficient GLA kernel
