# Concept: valid audio tokens condition causal text

The variable-duration encoder returns:

```text
audio tokens:    (batch, max_audio_tokens, embed_dim)
audio validity:  (batch, max_audio_tokens)
```

The language model constructs:

```text
[padded audio tokens] [separator] [text tokens]
```

## Attention structure

Valid audio tokens and the separator form a bidirectional observed prefix.
Invalid audio queries and keys are disabled. Text queries can inspect every valid
prefix token and causal text through their own position, but prefix queries never
inspect text.

Every text position is valid in this lesson because prompts are equal-length.
The earlier multimodal SFT lesson covers padded text batches and ignored labels.

## Carrying mechanisms forward

Greedy generation and evaluation are modality-independent once the model accepts
an observed prefix plus text. Their complete implementations are provided from
`llm.24_tiny_video_language_model`. The learner should inspect that code, but not
rewrite it merely because the prefix is now audio.

Generation recomputes the tiny sequence at every step. KV-cache prefill and
decoding were covered in `llm.17_multimodal_generation`.
