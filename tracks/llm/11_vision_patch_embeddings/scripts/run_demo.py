"""Trace a numbered image from pixels to patches to visual tokens."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import VisionPatchEmbedding, patchify, unpatchify
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import VisionPatchEmbedding, patchify, unpatchify


def main() -> None:
    torch.manual_seed(11)
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    patches = patchify(image, patch_size=2)
    reconstructed = unpatchify(
        patches,
        image_height=4,
        image_width=4,
        channels=1,
        patch_size=2,
    )
    embedder = VisionPatchEmbedding(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        embed_dim=6,
    )
    tokens = embedder(image)

    print("numbered image:")
    print(image[0, 0].to(dtype=torch.int64))
    print("\nrow-major flattened patches:")
    print(patches[0].to(dtype=torch.int64))
    print(f"\nimage shape:         {tuple(image.shape)}")
    print(f"patch sequence shape: {tuple(patches.shape)}")
    print(f"visual token shape:   {tuple(tokens.shape)}")
    print(f"patch grid:           {embedder.grid_height} x {embedder.grid_width}")
    print(f"round-trip exact:     {torch.equal(image, reconstructed)}")
    print("Each patch is now one positioned token ready for visual attention.")


if __name__ == "__main__":
    main()
