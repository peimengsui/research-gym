# Concept: diffuse tokens into an absorbing mask state

## Continuous noise versus discrete corruption

Gaussian diffusion gradually reduces the signal in a continuous sample. Token
IDs cannot be meaningfully perturbed by adding a small real number. A masked
discrete process instead replaces ordinary tokens with a dedicated mask token.

For cumulative masking probability `m_t`, every eligible clean token has the
direct marginal:

```text
q(x_t | x_0):
    x_t = MASK  with probability m_t
    x_t = x_0   with probability 1 - m_t
```

This lesson uses a linear cumulative schedule from zero to one. Timestep zero
is fully clean; the final timestep masks every eligible token.

## Which positions may be corrupted?

Only ordinary content tokens are eligible. Three masks define that set:

```text
eligible = attention_mask
           AND NOT special_token_mask
           AND input_id != mask_token_id
```

The attention mask excludes padding. The special-token mask preserves markers
such as BOS and EOS. Excluding an already-present mask token makes the absorbing
state explicit.

## Direct noising

For one timestep per sequence, gather one masking probability and compare it
with independent uniform random values at every position:

```text
corruption_mask = eligible AND (uniform < masking_probability)
corrupted_ids = where(corruption_mask, MASK, clean_ids)
```

`corruption_mask` also becomes the denoising loss mask. A model should predict
the original token only where information was removed; unchanged content,
special tokens, and padding do not contribute to this lesson's objective.

## Empty loss masks are valid

At timestep zero, or occasionally at a low-noise timestep, a batch can contain
no corrupted positions. That is not malformed data. The masked cross-entropy
returns a graph-connected zero (`logits.sum() * 0`) so a training loop may
safely backpropagate and continue.

## Direct marginals versus one absorbing trajectory

Calling the direct sampler independently at several timesteps gives correct
marginals, but newly drawn randomness could make a token appear unmasked after
it was masked in an earlier visualization.

To couple those marginals into one absorbing trajectory, draw one uniform value
per token and reuse it at every timestep:

```text
token is masked at t exactly when uniform < m_t
```

Because `m_t` is nondecreasing, a masked token can never return to its original
state. The shared random thresholds expose the absorbing path without requiring
a separate step-by-step transition kernel.

## Shapes

```text
clean/corrupted token IDs: [batch, sequence]
attention/special masks:   [batch, sequence]
corruption/loss masks:     [batch, sequence]
timesteps/probabilities:   [batch]
model logits:              [batch, sequence, vocabulary]
trajectory:                [num_timesteps, batch, sequence]
```
