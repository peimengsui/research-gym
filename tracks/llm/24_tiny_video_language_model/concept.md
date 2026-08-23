# Concept: an observed video prefix conditions causal text

The factorized video encoder produces `(B, V, D)` tokens. The language model
constructs one sequence:

```text
[video tubelet tokens] [separator] [text tokens]
```

The video and separator form an observed prefix. Prefix queries communicate
bidirectionally inside that prefix. Every text query can inspect the full prefix
and earlier text, but never future text.

## Training

At text position `i`, logits predict text token `i + 1`. The final text position
has no target and receives `IGNORE_INDEX`. Cross-entropy therefore trains only
valid next-token decisions while the video tokens provide context.

## Generation

Greedy generation repeatedly:

1. runs the model on video plus all text generated so far
2. reads logits at the final text position
3. appends the highest-logit token
4. stops when all rows emit EOS or the token budget is exhausted

Finished rows append EOS so the batch remains rectangular. This simple lesson
recomputes the video and text sequence each step. `llm.17_multimodal_generation`
and `llm.06_kv_cache` show how to prefill once and reuse cached keys/values.

## Evaluation

Free generation tests the actual decoding path but can be brittle: a valid
paraphrase may fail exact match. Candidate ranking uses teacher forcing and the
mean log probability of each answer, making model comparisons deterministic but
constrained to supplied choices. The tiny harness reports both.

This is an educational architecture, not a claim that fixed tubelets and exact
match are sufficient for realistic video understanding.
