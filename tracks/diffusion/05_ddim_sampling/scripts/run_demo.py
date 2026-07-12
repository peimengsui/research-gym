"""Show deterministic DDIM sampling with an oracle noise predictor."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        OracleNoisePredictor,
        ddim_sample_loop,
        make_diffusion_schedule,
        q_sample,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py",
            WORKSPACE_ROOT / "implementation.py",
        )
    from implementation import (
        OracleNoisePredictor,
        ddim_sample_loop,
        make_diffusion_schedule,
        q_sample,
    )


def make_square_image() -> torch.Tensor:
    """Create one tiny image in [-1, 1] with shape (1, 1, 8, 8)."""

    image = torch.full((1, 1, 8, 8), -1.0)
    image[:, :, 2:6, 2:6] = 1.0
    return image


def mean_absolute_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean(torch.abs(a - b)).item()


def main() -> None:
    torch.manual_seed(5)
    schedule = make_diffusion_schedule(
        num_timesteps=12,
        beta_start=0.01,
        beta_end=0.12,
    )
    x_start = make_square_image()
    final_timestep = torch.tensor([schedule.num_timesteps - 1])
    x_noisy, _ = q_sample(
        schedule,
        x_start,
        final_timestep,
        noise=torch.randn_like(x_start),
    )
    oracle = OracleNoisePredictor(schedule, x_start)

    full_steps = ddim_sample_loop(
        oracle,
        schedule,
        shape=tuple(x_start.shape),
        num_steps=schedule.num_timesteps,
        eta=0.0,
        initial_noise=x_noisy,
    )
    few_steps = ddim_sample_loop(
        oracle,
        schedule,
        shape=tuple(x_start.shape),
        num_steps=4,
        eta=0.0,
        initial_noise=x_noisy,
    )

    print(f"clean image shape:          {tuple(x_start.shape)}")
    print(f"starting noisy mean:        {x_noisy.mean().item(): .4f}")
    print(f"full-step DDIM MAE:         {mean_absolute_error(full_steps, x_start):.6f}")
    print(f"four-step DDIM MAE:         {mean_absolute_error(few_steps, x_start):.6f}")
    print("With eta=0 and an oracle epsilon model, DDIM follows a deterministic path.")


if __name__ == "__main__":
    main()
