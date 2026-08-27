# Guide

Open `implementation.py` and complete five TODOs.

## 1. Compute a log-magnitude spectrogram

Create a Hann window on the waveform device and dtype. Call `torch.stft` with
`center=False` and `return_complex=True`, then apply `log1p` to its magnitude.

## 2–3. Patch and reconstruct

Unfold frequency and time with non-overlapping patch strides. Rearrange the grid
to `(batch, time_grid, frequency_grid, frequency_patch, time_patch)` before
flattening. Reconstruction reverses exactly those operations.

## 4–5. Embed content and position

Create a projection plus `temporal_position_embedding` and
`frequency_position_embedding`. Keep these names so the factorization is easy to
inspect. Temporal IDs repeat for every frequency patch; the complete frequency
range repeats at every time step.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include enabling centered STFT padding, confusing frequency/time
axes, flattening in frequency-major order, and creating the Hann window on a
different device or with a different dtype than the waveform.
