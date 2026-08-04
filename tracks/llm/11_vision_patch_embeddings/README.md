# Images as Patch Tokens

Text Transformers receive a sequence of token vectors. A vision Transformer can
use the same interface after dividing an image into patches and mapping each
patch to an embedding vector.

In this lesson you will implement:

- row-major image patch extraction with explicit tensor shapes
- exact reconstruction from flattened patches
- a shared linear projection from patch pixels to token embeddings
- learned positional embeddings for the two-dimensional patch grid
- a small `VisionPatchEmbedding` module that returns Transformer-ready tokens

This is a native vision implementation: it uses only PyTorch tensor operations,
not a pretrained or hidden vision encoder. Attention is intentionally left for
`llm.12_vision_attention`.

## Start

```bash
uv run rgym start llm.11_vision_patch_embeddings
cd workspace/llm.11_vision_patch_embeddings
uv run rgym test
uv run rgym run
```
