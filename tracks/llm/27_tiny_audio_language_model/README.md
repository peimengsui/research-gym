# Audio-Text Generation and Evaluation

This lesson completes the native-audio sequence by using variable-duration audio
tokens as a prefix for causal language modeling.

The audio encoder, masked Transformer blocks, input validation, generation loop,
candidate scoring, and evaluation harness are carried forward in complete form.
You will focus on:

- a dense valid-audio-prefix / causal-text attention mask
- assembling audio, separator, and text embeddings
- producing text logits and next-token loss

The demo trains briefly on synthetic low/high tones and then exercises generation
and candidate ranking. It is a pipeline check, not an audio benchmark.

## Start

```bash
uv run rgym start llm.27_tiny_audio_language_model
cd workspace/llm.27_tiny_audio_language_model
uv run rgym test
uv run rgym run
```
