# Spatial and Temporal Attention

Full attention treats every video token as one long sequence. Factorized video
attention instead performs two smaller operations: spatial attention within each
time step, then temporal attention along each spatial location.

The tubelet embedding from `llm.22` is already complete in `provided.py`. You
will implement:

- readable multi-head self-attention
- spatial and temporal token reshaping
- a factorized pre-norm video Transformer block
- a tiny stack that returns flattened contextualized video tokens

## Start

```bash
uv run rgym start llm.23_factorized_video_attention
cd workspace/llm.23_factorized_video_attention
uv run rgym test
uv run rgym run
```
