# Review

After the tests pass, explain:

- why STFT creates both frequency and time axes
- how `n_fft` and `hop_length` change the representation
- why this lesson uses `log1p(abs(stft))`
- how a flat token index maps to time and frequency patches
- why patch reconstruction can be exact while waveform reconstruction cannot

Compare your work with `solution.locked.py`, focusing on axis order rather than
matching syntax.
