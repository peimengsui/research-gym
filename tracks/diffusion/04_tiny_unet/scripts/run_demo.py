"""Train a tiny U-Net to predict image-shaped diffusion noise."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        TinyUNet,
        image_noise_prediction_loss,
        make_diffusion_schedule,
        make_image_noise_prediction_batch,
        make_tiny_images,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py",
            WORKSPACE_ROOT / "implementation.py",
        )
    from implementation import (
        TinyUNet,
        image_noise_prediction_loss,
        make_diffusion_schedule,
        make_image_noise_prediction_batch,
        make_tiny_images,
    )


def main() -> None:
    torch.manual_seed(12)
    schedule = make_diffusion_schedule(
        num_timesteps=12,
        beta_start=0.01,
        beta_end=0.12,
    )
    images = make_tiny_images(n=16, image_size=8)
    generator = torch.Generator().manual_seed(13)
    batch = make_image_noise_prediction_batch(
        schedule,
        images,
        batch_size=8,
        generator=generator,
    )
    model = TinyUNet(in_channels=1, base_channels=8, time_embed_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    print(f"x_t shape:       {tuple(batch.x_t.shape)}")
    print(f"timesteps shape: {tuple(batch.timesteps.shape)}")
    print(f"noise shape:     {tuple(batch.noise.shape)}")
    print()

    with torch.no_grad():
        initial_loss = image_noise_prediction_loss(
            model, batch.x_t, batch.timesteps, batch.noise
        )
    print(f"initial image epsilon loss: {initial_loss.item():.4f}")

    for step in range(1, 81):
        loss = image_noise_prediction_loss(
            model, batch.x_t, batch.timesteps, batch.noise
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, 20, 40, 80}:
            print(f"step {step:03d} | loss={loss.item():.4f}")

    with torch.no_grad():
        final_loss = image_noise_prediction_loss(
            model, batch.x_t, batch.timesteps, batch.noise
        )
        predicted = model(batch.x_t[:1], batch.timesteps[:1])

    print(f"final image epsilon loss:   {final_loss.item():.4f}")
    print(f"example target noise mean:  {batch.noise[0].mean().item():.4f}")
    print(f"example predicted mean:     {predicted[0].mean().item():.4f}")
    print("The tiny U-Net learned to predict image-shaped Gaussian noise.")


if __name__ == "__main__":
    main()
