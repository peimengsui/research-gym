# Visual Prefix and Causal Text

Lesson 13 assembled `[visual, separator, text]` and returned a 1D validity mask.
This lesson converts that metadata into the 2D attention matrix a Transformer
actually needs.

In this lesson you will implement:

- prefix length as visual tokens plus one separator
- a shared visual-prefix / causal-text allow pattern
- padding broadcast into a batched `(batch, seq, seq)` matrix
- end-to-end construction from 1D validity metadata
- masked softmax that zeros padding query rows

## Start

```bash
uv run rgym start llm.14_multimodal_attention_mask
cd workspace/llm.14_multimodal_attention_mask
uv run rgym test
uv run rgym run
```
