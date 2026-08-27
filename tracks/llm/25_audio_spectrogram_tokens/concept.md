# Concept: from samples to time-frequency tokens

A waveform has shape `(batch, samples)`. It shows amplitude over time but does
not explicitly separate frequencies.

## Short-time Fourier transform

The STFT slides a window over the waveform and computes a Fourier transform in
each window. With `center=False`, `n_fft=N`, hop size `H`, and `S` samples:

```text
frequency bins = floor(N / 2) + 1
time frames = 1 + floor((S - N) / H)
```

`torch.stft(..., return_complex=True)` returns complex values. This lesson uses:

```text
log_magnitude = log(1 + abs(complex_spectrogram))
```

The logarithm compresses the large range between quiet and strong components.

## Spectrogram patches

The spectrogram is `(B, frequency_bins, time_frames)`. A patch covers several
neighboring frequency bins and time frames. Tokens use time-major order: all
frequency patches in the earliest time window come first.

Each token receives:

```text
projected patch + temporal position + frequency position
```

Factorized positions preserve the two spectrogram axes and prepare the next
lesson to reason about valid time tokens in padded audio batches.

This representation discards phase. It is intended for learning audio features,
not exact waveform reconstruction.
