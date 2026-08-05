# Visual Transformer Blocks

Patch embeddings turn an image into a sequence, but each token initially
describes only one local patch and its position. Visual self-attention lets each
patch gather information from every other patch.

In this lesson you will implement:

- reshaping an embedding into multiple attention heads
- merging attention heads back into one embedding
- bidirectional scaled dot-product self-attention
- attention maps with explicit query and source patch axes
- a pre-norm visual Transformer block
- a tiny visual encoder built from patch embeddings and stacked blocks

The completed patch-embedding implementation from
`llm.11_vision_patch_embeddings` is included in the scaffold. Unlike GPT
attention, visual attention uses no causal mask because an image is available
all at once.

## Start

```bash
uv run rgym start llm.12_vision_attention
cd workspace/llm.12_vision_attention
uv run rgym test
uv run rgym run
```
