# Waveforms, STFT, and Audio Patches

Audio begins as a one-dimensional waveform. This lesson uses the short-time
Fourier transform (STFT) to expose frequency content over time, then groups the
resulting spectrogram into tokens for a Transformer.

You will implement:

- log-magnitude spectrograms with `torch.stft`
- time-major spectrogram patch extraction
- exact spectrogram reconstruction from patches
- linear projection with temporal and frequency position embeddings

All examples are generated directly with PyTorch. No audio files, datasets, or
additional audio dependency are required.

## Start

```bash
uv run rgym start llm.25_audio_spectrogram_tokens
cd workspace/llm.25_audio_spectrogram_tokens
uv run rgym test
uv run rgym run
```
