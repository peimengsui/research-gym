"""Inspect STFT peaks, spectrogram patches, and audio token embeddings."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import AudioSpectrogramEmbedding, waveform_to_log_spectrogram
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import AudioSpectrogramEmbedding, waveform_to_log_spectrogram


def main() -> None:
    sample_index = torch.arange(40, dtype=torch.float32)
    low = torch.sin(2 * torch.pi * 2 * sample_index / 16)
    high = torch.sin(2 * torch.pi * 4 * sample_index / 16)
    waveforms = torch.stack((low, high))
    spectrograms = waveform_to_log_spectrogram(waveforms, n_fft=16, hop_length=8)
    embedding = AudioSpectrogramEmbedding(40, 16, 8, 3, 2, 12)
    tokens = embedding(waveforms)

    peak_bins = spectrograms.mean(dim=-1).argmax(dim=-1)
    print(f"waveform shape:             {tuple(waveforms.shape)}")
    print(f"spectrogram shape:          {tuple(spectrograms.shape)}")
    print(f"peak frequency bins:        {peak_bins.tolist()}")
    print(f"embedded audio shape:       {tuple(tokens.shape)}")
    print(f"temporal token count:       {embedding.temporal_token_count}")
    print(f"frequency tokens per time: {embedding.frequency_token_count}")
    print("Tokens are ordered by time window, then frequency region.")


if __name__ == "__main__":
    main()
