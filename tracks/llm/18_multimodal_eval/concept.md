# Concept: evaluating a tiny vision-language model

Evaluation asks a different question from training: what behavior does the
model exhibit on examples that we can inspect? This lesson uses two views of
the same image-grounded question.

## Free-generation exact match

The model receives an image and prompt, then generates without seeing the
reference answer. The returned tensor still includes the prompt:

```text
[prompt..., generated answer..., <eos>, optional EOS padding...]
```

Evaluation removes the known prompt and stops before the first EOS. Exact match
is one only when the remaining token IDs equal the reference IDs exactly.

Exact match is clear but strict. A semantically valid paraphrase counts as
wrong. That limitation is useful to see explicitly in a tiny harness.

## Teacher-forced candidate ranking

Some tasks provide a small answer set. For each candidate, concatenate:

```text
[prompt tokens, candidate tokens]
```

Run the full sequence and read the probability assigned to each candidate token
by its preceding position. If the prompt has length `P`, the first answer token
is predicted by logits at `P - 1`, the final prompt position—not logits at `P`.

For candidate tokens `a_1 ... a_M`, the score is:

```text
(log p(a_1 | image, prompt) + ... + log p(a_M | earlier context)) / M
```

Using a mean reduces the built-in preference of summed log probabilities for
shorter candidates. It is still a modeling choice worth reporting.

## EOS belongs in candidate scoring

References omit EOS so generated answers remain readable. Candidate sequences
may include EOS because stopping at the correct point is part of model quality.
Before comparing the selected candidate with its reference, strip EOS again.

## Aggregate and inspect

A single accuracy number hides failure modes. The harness therefore returns
both aggregate metrics and records containing generated IDs, reference IDs,
candidate scores, and the selected candidate. Small records make wrong answers
traceable rather than merely countable.

## Evaluation mode and gradients

Evaluation runs under `torch.no_grad()` and temporarily calls `model.eval()`.
The harness restores whether the caller originally had the model in training or
evaluation mode. This matters once models contain dropout or other mode-dependent
layers, even though this tiny model does not.
