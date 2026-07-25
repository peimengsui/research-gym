"""Show the same initial noise following two different token contexts."""

import shutil
import sys
from pathlib import Path

import torch
from torch import nn

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        TinyAutoencoder,
        TokenContextEncoder,
        gather_by_timestep,
        make_diffusion_schedule,
        sample_conditioned_images,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        TinyAutoencoder,
        TokenContextEncoder,
        gather_by_timestep,
        make_diffusion_schedule,
        sample_conditioned_images,
    )


class PromptOracle(nn.Module):
    def __init__(self, schedule, vertical: torch.Tensor, horizontal: torch.Tensor):
        super().__init__()
        self.schedule = schedule
        self.register_buffer("vertical", vertical)
        self.register_buffer("horizontal", horizontal)

    def forward(self, z_t, timesteps, context, context_mask=None):
        horizontal_weight = context[:, 0, 0].reshape(-1, 1, 1, 1)
        clean = (
            1.0 - horizontal_weight
        ) * self.vertical + horizontal_weight * self.horizontal
        signal = gather_by_timestep(self.schedule.sqrt_alpha_bars, timesteps, z_t.shape)
        noise_scale = gather_by_timestep(
            self.schedule.sqrt_one_minus_alpha_bars, timesteps, z_t.shape
        )
        return (z_t - signal * clean) / noise_scale


def main() -> None:
    torch.manual_seed(8)
    autoencoder = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    context_encoder = TokenContextEncoder(vocab_size=4, max_length=2, context_dim=4)
    with torch.no_grad():
        context_encoder.token_embedding.weight.zero_()
        context_encoder.position_embedding.weight.zero_()
        context_encoder.token_embedding.weight[2, 0] = 1.0

    vertical = torch.zeros(1, 2, 4, 4)
    vertical[:, :, :, 1:3] = 1.0
    horizontal = torch.zeros(1, 2, 4, 4)
    horizontal[:, :, 1:3, :] = 1.0
    schedule = make_diffusion_schedule(12, beta_start=0.01, beta_end=0.12)
    oracle = PromptOracle(schedule, vertical, horizontal)
    prompts = torch.tensor([[1], [2]])
    shared_noise = torch.randn(1, 2, 4, 4).expand(2, -1, -1, -1).clone()

    images, latents = sample_conditioned_images(
        autoencoder,
        context_encoder,
        oracle,
        schedule,
        image_shape=(2, 1, 8, 8),
        token_ids=prompts,
        context_mask=torch.ones_like(prompts, dtype=torch.bool),
        latent_scale=1.0,
        num_steps=5,
        initial_noise=shared_noise,
    )

    vertical_mae = torch.mean(torch.abs(latents[0] - vertical[0])).item()
    horizontal_mae = torch.mean(torch.abs(latents[1] - horizontal[0])).item()
    separation = torch.mean(torch.abs(latents[0] - latents[1])).item()
    print(f"shared initial noise:      {torch.equal(shared_noise[0], shared_noise[1])}")
    print("prompt 1 target:           vertical latent bar")
    print("prompt 2 target:           horizontal latent bar")
    print(f"prompt 1 latent MAE:       {vertical_mae:.6f}")
    print(f"prompt 2 latent MAE:       {horizontal_mae:.6f}")
    print(f"conditioned separation:    {separation:.4f}")
    print(f"decoded image shape:       {tuple(images.shape)}")
    print("The prompt context changed the result even though initial noise was shared.")


if __name__ == "__main__":
    main()
