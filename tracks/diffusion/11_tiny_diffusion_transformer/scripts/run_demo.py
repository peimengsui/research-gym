"""Train a tiny DiT briefly on fixed noisy latent grids."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    TinyDiffusionTransformer,
    diffusion_transformer_loss,
    patchify,
    unpatchify,
)
from provided import make_diffusion_schedule, make_tiny_latents  # noqa: E402


def main() -> None:
    torch.manual_seed(11)
    clean_samples = make_tiny_latents(n=4)
    schedule = make_diffusion_schedule(10)
    timesteps = torch.tensor([1, 3, 6, 9])
    noise = torch.randn_like(clean_samples)
    model = TinyDiffusionTransformer(
        embedding_dim=16,
        depth=1,
        num_heads=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    patches = patchify(clean_samples, patch_size=2)
    reconstructed = unpatchify(patches, 2, 8, 8, 2)
    losses = []
    for _ in range(10):
        loss = diffusion_transformer_loss(
            model,
            schedule,
            clean_samples,
            "epsilon",
            timesteps,
            noise,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"latent shape:             {tuple(clean_samples.shape)}")
    print(f"patch sequence shape:     {tuple(patches.shape)}")
    print(f"patch round-trip exact:   {torch.equal(reconstructed, clean_samples)}")
    print(f"initial denoising loss:   {losses[0]:.6f}")
    print(f"final denoising loss:     {losses[-1]:.6f}")
    print("The transformer predicts one denoising target patch for every input patch.")


if __name__ == "__main__":
    main()
