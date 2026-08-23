"""Show the two attention axes used by a tiny video encoder."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import TinyVideoEncoder
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import TinyVideoEncoder


def main() -> None:
    torch.manual_seed(23)
    videos = torch.zeros(2, 4, 1, 4, 4)
    videos[0, :, :, :, :2] = 1.0
    videos[1, :, :, :, 2:] = 1.0
    encoder = TinyVideoEncoder(4, 4, 2, 2, 1, 8, 2, 2)
    tokens, maps = encoder(videos, return_weights=True)
    spatial, temporal = maps[0]

    full_attention_entries = (2 * 4) ** 2
    factorized_entries = 2 * (4**2) + 4 * (2**2)
    print(f"video batch shape:            {tuple(videos.shape)}")
    print(f"encoder output shape:         {tuple(tokens.shape)}")
    print(f"first spatial weights shape:  {tuple(spatial.shape)}")
    print(f"first temporal weights shape: {tuple(temporal.shape)}")
    print(f"full attention entries:       {full_attention_entries}")
    print(f"factorized entries:           {factorized_entries}")
    print("Both passes are bidirectional because the full input clip is observed.")


if __name__ == "__main__":
    main()
