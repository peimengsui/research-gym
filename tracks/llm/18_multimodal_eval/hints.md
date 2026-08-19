# Hints

## Hint 1

The generated continuation is `generated_token_ids[prompt_length:]`. Pass that
slice to `_tokens_before_eos`.

## Hint 2

For prompt length `P` and candidate length `C`, use the model logits slice
`logits[P - 1 : P - 1 + C]`.

## Hint 3

After `F.log_softmax(candidate_logits, dim=-1)`, gather with
`candidate_token_ids.unsqueeze(1)` and average the selected values.

## Hint 4

Python's `max(range(len(scores)), key=scores.__getitem__)` returns the index of
the highest score while preserving the original score order.

## Hint 5

Store `was_training = model.training`, call `model.eval()`, and restore with
`model.train(was_training)` inside `finally`.

## Hint 6

Booleans sum like zero/one values. Divide each sum by `len(records)` to obtain
the two accuracies.
