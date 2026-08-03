# Language Model Sampling

A language model produces logits, not text. Sampling policy determines how
those logits become generated token IDs and controls the trade-off between
determinism, diversity, and low-probability mistakes.

In this lesson you will implement:

- temperature scaling
- top-k candidate filtering
- top-p, or nucleus, filtering
- greedy and stochastic token selection
- autoregressive generation with context cropping
- optional end-of-sequence stopping

Tiny GPT is included as working code. Your new work is entirely in the decoding
policy that sits after the model's final-token logits.

## Start

```bash
uv run rgym start llm.10_sampling
cd workspace/llm.10_sampling
uv run rgym test
uv run rgym run
```
