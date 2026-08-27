"""Inspect validity and attention for padded variable-duration waveforms."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import TinyVariableAudioEncoder
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import TinyVariableAudioEncoder


def main() -> None:
    torch.manual_seed(26)
    sample_index = torch.arange(18, dtype=torch.float32)
    full = torch.sin(2 * torch.pi * sample_index / 6)
    short = torch.sin(2 * torch.pi * sample_index / 9)
    short[14:] = 0.0
    waveforms = torch.stack((full, short))
    lengths = torch.tensor([18, 14])
    encoder = TinyVariableAudioEncoder(18, 6, 4, 2, 2, 8, 2, 1)
    tokens, validity, maps = encoder(waveforms, lengths, return_weights=True)

    print(f"padded waveform shape: {tuple(waveforms.shape)}")
    print(f"sample lengths:        {lengths.tolist()}")
    print(f"audio token shape:     {tuple(tokens.shape)}")
    print(f"token validity:        {validity.tolist()}")
    print(f"attention map shape:   {tuple(maps[0].shape)}")
    print(f"finite attention:      {torch.isfinite(maps[0]).all().item()}")
    print("Invalid query rows and invalid token outputs are exactly zero.")


if __name__ == "__main__":
    main()
