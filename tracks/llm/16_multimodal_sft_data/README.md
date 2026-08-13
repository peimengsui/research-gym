# Image-Text Conversations

The tiny native VLM can already predict text from image and text context. This
lesson turns image-grounded conversations into supervised fine-tuning batches
that train only on assistant responses.

You will implement:

- explicit user and assistant conversation formatting
- assistant-only next-token label masks
- truncation before causal shifting
- fixed-width right-padded text batches
- same-shape synthetic image collation
- one SFT loss call through the completed native VLM

`provided.py` contains the completed lesson 15 model. The exercise is about the
data and supervision contract, not new model architecture.

## Start

```bash
uv run rgym start llm.16_multimodal_sft_data
cd workspace/llm.16_multimodal_sft_data
uv run rgym test
uv run rgym run
```
