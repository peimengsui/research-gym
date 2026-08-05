"""Inspect full visual attention across a four-patch image."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import TinyVisionEncoder
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import TinyVisionEncoder


def main() -> None:
    torch.manual_seed(12)
    image = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [2.0, 2.0, 3.0, 3.0],
        ]
    ).reshape(1, 1, 4, 4)
    encoder = TinyVisionEncoder(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
    )

    tokens, attention_maps = encoder(image, return_weights=True)
    weights = attention_maps[0]

    print("image with four constant 2 x 2 patches:")
    print(image[0, 0])
    print(f"\nvisual token shape: {tuple(tokens.shape)}")
    print(f"attention shape:    {tuple(weights.shape)}")
    print("\nhead 0 attention (rows=query patch, columns=source patch):")
    print(weights[0, 0].detach().round(decimals=3))
    print(f"\nrow sums: {weights[0, 0].sum(dim=-1).detach()}")
    print("Every matrix entry is available: visual attention has no causal mask.")


if __name__ == "__main__":
    main()
