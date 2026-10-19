# Concept: text generation by iterative unmasking

An autoregressive language model factorizes a sequence from left to right. At
generation time it appends one token after another, so each prediction can use
only the prefix already generated.

A masked discrete diffusion model uses a different path. Its forward process
gradually replaces ordinary tokens with one absorbing `[MASK]` token. A learned
reverse model sees a partially masked sequence and predicts the clean token at
every position:

```text
[BOS] [MASK] blue [MASK] [EOS]
             |
             v
       bidirectional denoiser
             |
             v
       red    blue  bird
```

Because the denoiser is bidirectional, a masked position can use visible
tokens on both its left and right. The model also receives the current
timestep, which tells it how corrupted the input is expected to be.

## Training objective

For a clean sequence `x_0`, sample a timestep `t`, use the forward process to
obtain `x_t`, and predict `x_0` from `(x_t, t)`. Cross-entropy is evaluated only
where the forward process actually replaced an eligible token:

```text
loss = mean CE(model(x_t, t)[corruption_mask], x_0[corruption_mask])
```

Padding and structural tokens such as `[BOS]` and `[EOS]` are never corruption
targets. An empty corruption mask returns a differentiable zero rather than
inventing supervision.

## Reverse process

Generation begins with every editable content slot masked. At each reverse
step the model predicts an ordinary token and confidence for every remaining
mask. The sampler fixes a subset of the highest-confidence predictions, then
runs the model again on the updated sequence.

```text
[BOS] [MASK] [MASK] [MASK] [EOS]
[BOS] [MASK] blue   [MASK] [EOS]
[BOS] red    blue   [MASK] [EOS]
[BOS] red    blue   bird   [EOS]
```

The number revealed at each step is chosen so all editable masks are filled by
the end. Once a token is fixed, it is never revised in this small lesson. Some
production systems use remasking or stochastic sampling, but confidence-based
monotonic unmasking makes the core algorithm easy to inspect.

## Structural invariants

The sampler may edit a position only when all three conditions are true:

```text
token is [MASK]
and attention_mask is True
and special_token_mask is False
```

This single rule protects padding, special tokens, and prompt tokens that were
already fixed. Output filtering also prevents `[PAD]`, `[BOS]`, `[EOS]`, and
`[MASK]` from being selected as generated content.
