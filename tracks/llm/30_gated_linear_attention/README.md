# Gated Linear Attention and Selective Memory

Build a compact, readable version of Gated Linear Attention (GLA): a modern
linear recurrent layer whose learned, data-dependent forget gates decide which
parts of a matrix-valued memory to retain.

## What you will build

- numerically stable log-space decay gates
- the recurrent GLA matrix-state update
- an equivalent explicit parallel formulation
- low-rank gate projection, per-head RMS normalization, and a SiLU output gate
- full-sequence and stateful chunk processing with the same outputs

The lesson follows the key-only gating simplification used in
[Gated Linear Attention Transformers with Hardware-Efficient Training](https://proceedings.mlr.press/v235/yang24ab.html)
and reflected in the
[official Flash Linear Attention implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gla.py).

The explicit parallel function is a small CPU-friendly correctness oracle. It
materializes pairwise decay factors and is not the paper's fused, tiled training
kernel. Production GLA obtains hardware efficiency with chunkwise algorithms;
implementing Triton kernels is outside this lesson.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
