"""Inspect tubelet order and factorized video position embeddings."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import VideoTubeletEmbedding, tubeletify, untubeletify
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import VideoTubeletEmbedding, tubeletify, untubeletify


def main() -> None:
    torch.manual_seed(22)
    video = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 1, 4, 4)
    tubelets = tubeletify(video, tubelet_size=2, patch_size=2)
    reconstructed = untubeletify(tubelets, 4, 4, 4, 1, 2, 2)
    embedder = VideoTubeletEmbedding(4, 4, 4, 2, 2, 1, 8)
    tokens = embedder(video)

    print(f"video shape:             {tuple(video.shape)}")
    print(f"tubelet shape:           {tuple(tubelets.shape)}")
    print(f"embedded token shape:    {tuple(tokens.shape)}")
    print(f"temporal token count:    {embedder.temporal_token_count}")
    print(f"spatial tokens per time: {embedder.spatial_token_count}")
    print(f"first tubelet values:    {tubelets[0, 0].tolist()}")
    print(f"exact reconstruction:    {torch.equal(video, reconstructed)}")


if __name__ == "__main__":
    main()
