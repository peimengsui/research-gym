# Hints

1. A real-valued STFT has `n_fft // 2 + 1` frequency bins.
2. After two unfolds, inspect axis order before calling `permute`.
3. Time-major IDs use `repeat_interleave(frequency_token_count)`.
4. Frequency IDs use `repeat(temporal_token_count)`.
5. Reconstruct the spectrogram, not the waveform; phase was intentionally
   discarded.
