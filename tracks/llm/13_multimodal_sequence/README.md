# Unified Image and Text Tokens

A native multimodal Transformer needs image and text representations to meet in
one sequence and one embedding space. This lesson builds that interface without
yet applying multimodal attention.

In this lesson you will implement:

- text padding masks
- a unified validity mask for visual, separator, and text positions
- token-type IDs that preserve modality boundaries
- text, separator, token-type, and full-sequence position embeddings
- deterministic `[visual tokens, separator, text tokens]` concatenation
- a structured result that carries embeddings and mask metadata together

The native visual encoder from lessons 11 and 12 is complete in `provided.py`.
The learner does not rewrite patch extraction or visual attention.

## Start

```bash
uv run rgym start llm.13_multimodal_sequence
cd workspace/llm.13_multimodal_sequence
uv run rgym test
uv run rgym run
```
