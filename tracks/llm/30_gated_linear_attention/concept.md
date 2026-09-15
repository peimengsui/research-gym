# Concept: learned forgetting in a matrix memory

## From linear attention to GLA

Linear recurrent attention stores a matrix rather than every past key and
value. For one attention head, a simple state update is:

```text
S_t = S_{t-1} + k_t v_t^T
o_t = scale * q_t S_t
```

The state `S_t` has shape `[key_dim, value_dim]`. Its size stays constant as the
sequence grows, but every write persists forever. New tokens cannot selectively
clear stale information.

[Yang et al. (2024)](https://proceedings.mlr.press/v235/yang24ab.html) introduce
data-dependent forget gates. Their general update uses a two-dimensional gate:

```text
S_t = G_t ⊙ S_{t-1} + k_t v_t^T
```

The paper factorizes `G_t` for efficiency and finds that key-side gating is a
useful simplification. This lesson implements that practical form:

```text
S_t = alpha_t[:, None] ⊙ S_{t-1} + k_t v_t^T
o_t = scale * q_t S_t
```

Here `alpha_t` has one decay value per key channel. A value near one retains
that row of memory; a value near zero forgets it before the current key/value
pair is written.

## Stable gate parameterization

The model predicts unconstrained gate logits with a low-rank projection. We
store log decays:

```text
log(alpha_t) = logsigmoid(gate_logits_t) / normalizer
alpha_t      = exp(log(alpha_t))
```

This guarantees `0 < alpha_t < 1`. The normalizer controls the default memory
timescale: larger values move the decay closer to one. Log-space values are
also what parallel GLA kernels use to accumulate many decay products more
stably.

The low-rank gate path is:

```text
x -> gate_down -> gate_up -> [batch, heads, time, key_dim]
```

It makes the forget decision input-dependent without requiring a full separate
projection directly from the model width to every gate channel.

## Recurrent form

At every position:

```text
memory = exp(log_decay_t)[..., None] * memory
memory = memory + k_t[..., None] * v_t[..., None, :]
output = scale * q_t @ memory
```

The write happens before the read, so position `t` can use its own key and
value. Passing the final memory into a later chunk gives exact streaming
continuation with constant state size.

## Explicit parallel form

Unrolling the recurrence shows how much a write at source position `i` remains
when target position `t` reads it:

```text
retention(t, i) = product_{j=i+1..t} alpha_j    when i <= t
```

With cumulative log decays:

```text
log_retention(t, i) = cumulative_log_decay_t - cumulative_log_decay_i
```

The current write has retention one. Future writes are causally masked. These
retentions combine with query-key products to produce all outputs in parallel.

This lesson materializes `[time, time, key_dim]` retention factors so the
equivalence is transparent and testable. The GLA paper's hardware contribution
instead divides sequences into chunks, combines recurrent inter-chunk state
with parallel intra-chunk work, and avoids materializing this large tensor.

## Output path

The raw GLA read is not softmax-normalized. A practical GLA layer stabilizes and
controls it with:

```text
normalized = RMSNorm_per_head(raw_output)
gate       = SiLU(output_gate_projection(x))
y          = output_projection(normalized * gate)
```

The forget gate controls memory retention. The output gate separately controls
how much of the retrieved value enters the residual stream.

## Shapes

```text
query/key:   [batch, heads, time, key_dim]
value:       [batch, heads, time, value_dim]
log_decay:   [batch, heads, time, key_dim]
output:      [batch, heads, time, value_dim]
state.memory:[batch, heads, key_dim, value_dim]
```

## Deliberate lesson boundaries

To keep the implementation small and CPU-friendly, the lesson omits short
convolutions, grouped KV heads, separate key/value expansion ratios, affine
RMSNorm, padding/unpadding, and fused Triton kernels. Those are engineering
extensions around the core GLA recurrence taught here.
