# Hints

## Hint 1

Temperature changes relative logit gaps: `logits / temperature`.

## Hint 2

Initialize filtered logits with `torch.full_like(logits, float("-inf"))`.

## Hint 3

For top-p, shift the removal mask one position to the right after comparing the
cumulative probability with `p`. This retains the token that crosses the
threshold.

## Hint 4

After sorting and filtering, use `scatter` with the sorted vocabulary indices
to restore the original order.

## Hint 5

`torch.multinomial(probabilities, 1, generator=generator)` already returns
shape `(batch, 1)`.

## Hint 6

Crop only the model input. Do not crop the sequence that you return to the
learner.
