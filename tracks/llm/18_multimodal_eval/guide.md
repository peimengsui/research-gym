# Guide

Open `implementation.py`. Complete the four TODOs in order.

## 1. Extract a generated answer

Input is one 1D generated sequence containing both prompt and continuation.
Slice from `prompt_length`, then keep only tokens before the first EOS. The
provided helper handles EOS truncation and rectangular EOS padding.

## 2. Score one candidate

Concatenate prompt and candidate, add image/text batch dimensions, and call
`model.forward_full`. If the prompt length is `P`, slice candidate-predicting
logits starting at `P - 1` for exactly the candidate length.

Apply `F.log_softmax`, gather the log probability at each candidate token ID,
and return the mean. Candidate tensors may include EOS.

## 3. Select a candidate

Score candidates in their given order, convert scalar tensors with `.item()`,
and choose the index with the largest score. Return both the winning index and
all scores so later inspection is possible.

## 4. Aggregate evaluation

Remember `model.training`, switch to evaluation mode, and restore the original
mode in `finally`. For each example:

1. Generate from its image and prompt.
2. Extract the continuation before EOS.
3. Compare generated and reference token IDs.
4. Rank candidates and compare the selected answer after removing EOS.
5. Store a complete `EvaluationRecord`.

Average the two booleans separately for generation exact match and candidate
accuracy.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include comparing the prompt as part of the answer, retaining EOS in
only one side of a comparison, using logits at `P` instead of `P - 1`, summing
scores without documenting length bias, evaluating with gradients enabled, or
returning only aggregate metrics with no records to debug.
