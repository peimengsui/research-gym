# Variable-Duration Audio and Attention

Audio clips rarely share one duration. Batches pad them to a common sample
count, but padded samples must not become meaningful spectrogram tokens or
attention mass.

The STFT and patch embedder from `llm.25` are complete in `provided.py`. You will
implement:

- valid STFT frame counts from original sample lengths
- time-major audio-token validity masks
- safe masked softmax for fully invalid query rows
- multi-head audio self-attention
- a tiny variable-duration audio encoder

## Start

```bash
uv run rgym start llm.26_audio_temporal_attention
cd workspace/llm.26_audio_temporal_attention
uv run rgym test
uv run rgym run
```
