# Concept: sample padding becomes token validity

A padded waveform batch has shape `(B, max_samples)` plus original lengths
`(B,)`. With a non-centered STFT, a frame is valid only when its complete window
fits inside the original length:

```text
valid_frames = 0                                  if length < n_fft
valid_frames = 1 + floor((length - n_fft) / hop) otherwise
```

## From frames to patches

If one token spans `P` time frames, this lesson marks it valid only when all `P`
frames are valid:

```text
valid_time_tokens = floor(valid_frames / P)
```

This conservative rule prevents a patch from mixing real and padded samples.
Because tokens are time-major, each temporal validity flag repeats for every
frequency patch.

## Invalid attention rows

Invalid keys must receive no probability. Invalid queries should produce no
output. Masking every score in an invalid query row to negative infinity makes
ordinary softmax return NaNs, so the implementation converts those rows to zero
probabilities explicitly.

The encoder also zeros invalid residual positions after every block. Consequently,
changing padded waveform values cannot change valid encoded tokens.

This lesson uses bidirectional audio encoding: the full observed clip is context.
Causal masking will apply only to generated text in the next lesson.
