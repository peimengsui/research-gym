# Guide

Complete three TODO groups. The audio encoder, multimodal blocks, generation,
evaluation, and validation are provided.

## 1. Build the audio-prefix mask

Append a valid separator and valid text positions to audio validity. Prefix
queries see valid prefix keys only. Text queries see valid prefix keys plus
causal text. Finally require both query and key positions to be valid so padded
audio query rows are all false.

## 2. Assemble the sequence

Encode the padded waveforms, append the learned separator and text embeddings,
add unified positions, and return the embeddings, attention mask, and prefix
length. The prefix length includes all padded audio slots plus the separator;
validity controls which slots matter.

## 3. Produce text logits and loss

Pass the sequence through every provided Transformer block, normalize, slice
positions after the prefix, and project to vocabulary logits. If validated
targets are present, compute ignored-index cross-entropy.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include removing invalid audio slots and thereby making prefix
lengths differ across the batch, letting prefix queries inspect text, forgetting
the separator in `prefix_length`, and slicing logits from the wrong position.
