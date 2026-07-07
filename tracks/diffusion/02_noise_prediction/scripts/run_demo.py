"""Train a tiny MLP to predict the noise added by the forward process."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        TinyNoisePredictor,
        make_diffusion_schedule,
        make_noise_prediction_batch,
        make_toy_points,
        noise_prediction_loss,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py",
            WORKSPACE_ROOT / "implementation.py",
        )
    from implementation import (
        TinyNoisePredictor,
        make_diffusion_schedule,
        make_noise_prediction_batch,
        make_toy_points,
        noise_prediction_loss,
    )


def main() -> None:
    torch.manual_seed(7)
    schedule = make_diffusion_schedule(
        num_timesteps=16,
        beta_start=0.01,
        beta_end=0.12,
    )
    data = make_toy_points(64)
    generator = torch.Generator().manual_seed(8)
    batch = make_noise_prediction_batch(
        schedule,
        data,
        batch_size=96,
        generator=generator,
    )
    model = TinyNoisePredictor(data_dim=2, time_embed_dim=16, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)

    print(f"x_t shape:       {tuple(batch.x_t.shape)}")
    print(f"timesteps shape: {tuple(batch.timesteps.shape)}")
    print(f"noise shape:     {tuple(batch.noise.shape)}")
    print()

    with torch.no_grad():
        initial_loss = noise_prediction_loss(
            model,
            batch.x_t,
            batch.timesteps,
            batch.noise,
        )
    print(f"initial epsilon-prediction loss: {initial_loss.item():.4f}")

    for step in range(1, 181):
        loss = noise_prediction_loss(model, batch.x_t, batch.timesteps, batch.noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, 30, 90, 180}:
            print(f"step {step:03d} | loss={loss.item():.4f}")

    with torch.no_grad():
        final_loss = noise_prediction_loss(
            model,
            batch.x_t,
            batch.timesteps,
            batch.noise,
        )
        prediction = model(batch.x_t[:1], batch.timesteps[:1])

    print(f"final epsilon-prediction loss:   {final_loss.item():.4f}")
    print(f"example true noise:      {batch.noise[0].tolist()}")
    print(f"example predicted noise: {prediction[0].tolist()}")
    print("The model learned the supervised target used by DDPM training.")


if __name__ == "__main__":
    main()
