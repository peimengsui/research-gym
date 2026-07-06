"""Demonstrate the diffusion forward noising process on 2D points."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import make_diffusion_schedule, make_toy_points, q_sample
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py",
            WORKSPACE_ROOT / "implementation.py",
        )
    from implementation import make_diffusion_schedule, make_toy_points, q_sample


def main() -> None:
    torch.manual_seed(0)
    schedule = make_diffusion_schedule(
        num_timesteps=10,
        beta_start=0.02,
        beta_end=0.18,
    )
    x_start = make_toy_points(4)
    timesteps_to_show = [0, 3, 6, 9]

    print("Clean 2D points:")
    print(x_start.round(decimals=3))
    print()
    print("Forward process coefficients:")

    shared_noise = torch.randn_like(x_start)
    for timestep in timesteps_to_show:
        timesteps = torch.full((x_start.shape[0],), timestep, dtype=torch.long)
        x_t, _ = q_sample(schedule, x_start, timesteps, noise=shared_noise)
        signal = schedule.sqrt_alpha_bars[timestep].item()
        noise_scale = schedule.sqrt_one_minus_alpha_bars[timestep].item()
        print(
            f"t={timestep:02d} | signal={signal:.3f} "
            f"| noise={noise_scale:.3f} | first_point={x_t[0].tolist()}"
        )

    print()
    print("As t increases, signal shrinks and the same noise contributes more.")


if __name__ == "__main__":
    main()
