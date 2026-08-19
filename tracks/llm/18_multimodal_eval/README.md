# Tiny Vision-Language Evaluation Harness

A loss value alone does not show whether a vision-language model produces the
right answer from an image. This lesson builds a tiny, inspectable harness for
two complementary evaluation modes:

- free-generation exact match
- teacher-forced candidate ranking

You will implement answer extraction around prompt and EOS boundaries, align
candidate tokens with the logits that predict them, rank candidates by mean log
probability, and aggregate per-example records into metrics.

`provided.py` contains the completed cached VLM and generation loop from lesson
17. Synthetic dark/bright images and a fixed vocabulary keep tests self-contained
and CPU-friendly.

## Start

```bash
uv run rgym start llm.18_multimodal_eval
cd workspace/llm.18_multimodal_eval
uv run rgym test
uv run rgym run
```
