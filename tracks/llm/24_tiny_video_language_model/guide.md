# Guide

The factorized video encoder, head helpers, feed-forward layer, and toy
vocabulary are complete in `provided.py`. Complete seven TODO groups.
Input-shape, dtype, vocabulary-range, length, and evaluation-example validation
are also provided so the TODOs stay focused on the learning mechanisms.

## 1. Build the sequence mask

The prefix length is `video_token_count + 1` for the separator. Prefix queries
see only the full prefix. Text queries see the prefix plus text keys at or before
their own position. Return `(batch, total, total)` booleans.

## 2. Apply masked multimodal attention

Compute scaled dot-product attention. Expand the batch mask across heads, fill
disallowed scores with negative infinity, softmax over keys, and combine values.

## 3. Construct next-token targets

Shift text token IDs left. The final target is `IGNORE_INDEX`. This lesson uses
equal-length text batches; padding/data collation was covered in `llm.16`.

## 4. Connect video and text

Encode the video, append the learned separator and text embeddings, add unified
positions, build the mask, run every Transformer block, and produce logits only
at text positions. If the already validated targets are supplied, compute
cross-entropy.

## 5. Generate

Validation is provided. Repeatedly run the full model, choose `argmax`, append EOS
for already finished rows, and stop when every row is finished. The recomputation
is explicit educational simplicity, not an efficient serving path.

## 6–7. Score and evaluate

Candidate and example validation plus EOS stripping are provided. Concatenate
prompt and candidate; logits beginning at `prompt_length - 1` predict candidate
tokens. Average their selected log probabilities. Evaluation runs generation and
candidate ranking and restores the model's original training/evaluation mode.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include allowing prefix queries to inspect text, scoring a candidate
one position late, comparing EOS itself during exact match, and continuing to
sample content for rows that already finished.
