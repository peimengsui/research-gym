# Guide

The factorized video encoder, head helpers, feed-forward layer, and toy
vocabulary are complete in `provided.py`. Complete seven TODO groups.

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
at text positions. If targets are supplied, compute cross-entropy.

## 5. Generate

Validate the token budget, repeatedly run the full model, choose `argmax`, append
EOS for already finished rows, and stop when every row is finished. The
recomputation is explicit educational simplicity, not an efficient serving path.

## 6–7. Score and evaluate

For a candidate answer, concatenate prompt and candidate. Logits beginning at
`prompt_length - 1` predict candidate tokens. Average their selected log
probabilities. Evaluation runs generation and candidate ranking, strips EOS for
exact match, and restores the model's original training/evaluation mode.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include allowing prefix queries to inspect text, scoring a candidate
one position late, comparing EOS itself during exact match, and continuing to
sample content for rows that already finished.
