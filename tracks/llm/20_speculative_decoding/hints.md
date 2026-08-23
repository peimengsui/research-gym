# Hints

## Hint 1

Use `torch.clamp(p_token / q_token, max=1.0)` for acceptance.

## Hint 2

The correction is `residual / residual.sum()` after clamping `p - q` at zero.

## Hint 3

Retain each draft distribution before extending the context. Verification needs
the `q` that actually sampled each proposal.

## Hint 4

Draw acceptance noise with `torch.rand((), generator=generator)`.

## Hint 5

`target_model.score_draft(prefix, draft_tokens)` returns shape
`(draft_length + 1, vocab_size)`; the final row supplies the bonus token.

## Hint 6

When appending step outputs, check the token budget and EOS one token at a time.
