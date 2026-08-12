# Tiny Native Vision-Language Model

This lesson combines the native vision and multimodal foundations from lessons
11–14 into one end-to-end model. Images become patch tokens, text follows as a
causal region, and text-position hidden states predict the next text token.

You will implement:

- masked multi-head attention over the unified image-text sequence
- a pre-norm multimodal Transformer block
- shifted next-token targets with padding ignored
- a stack of multimodal blocks and a vocabulary head
- text-only logits and cross-entropy loss
- gradient flow through image, text, shared Transformer, and output paths

The completed image encoder, sequence builder, and mask helpers are available in
`provided.py`. No pretrained vision model is used.

## Start

```bash
uv run rgym start llm.15_tiny_native_vlm
cd workspace/llm.15_tiny_native_vlm
uv run rgym test
uv run rgym run
```
