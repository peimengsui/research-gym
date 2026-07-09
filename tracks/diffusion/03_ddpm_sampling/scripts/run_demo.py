"""Demonstrate reverse DDPM equations with an oracle noise predictor."""

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
        ddpm_reverse_step,
        make_diffusion_schedule,
        make_toy_points,
        predict_x0_from_noise,
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
        ddpm_reverse_step,
        make_diffusion_schedule,
        make_toy_points,
        predict_x0_from_noise,
        q_sample,
    )


def main() -> None:
    torch.manual_seed(11)
    schedule = make_diffusion_schedule(
        num_timesteps=8,
        beta_start=0.02,
        beta_end=0.16,
    )
    x_start = make_toy_points(1)
    start_timestep = schedule.num_timesteps - 1
    timesteps = torch.tensor([start_timestep])
    x_t, true_noise = q_sample(schedule, x_start, timesteps)
    oracle = OracleNoisePredictor(schedule, x_start)

    print(f"clean point:        {x_start[0].tolist()}")
    print(f"noisy point at t={start_timestep}: {x_t[0].tolist()}")
    print()

    predicted_x0 = predict_x0_from_noise(schedule, x_t, timesteps, true_noise)
    print(f"x0 estimate with true noise: {predicted_x0[0].tolist()}")
    print()

    current = x_t
    for timestep in reversed(range(schedule.num_timesteps)):
        t = torch.tensor([timestep])
        zero_noise = torch.zeros_like(current)
        current, mean = ddpm_reverse_step(
            oracle,
            schedule,
            current,
            t,
            noise=zero_noise,
        )
        distance = (current - x_start).norm().item()
        if timestep in {7, 4, 1, 0}:
            print(
                f"after reverse step t={timestep:02d}: "
                f"point={current[0].tolist()} | distance_to_clean={distance:.4f}"
            )

    print()
    print("The oracle is a debugging aid: it checks the sampler equations.")


if __name__ == "__main__":
    main()
