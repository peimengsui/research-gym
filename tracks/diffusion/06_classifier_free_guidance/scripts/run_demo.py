"""Demonstrate how classifier-free guidance changes a DDIM target."""

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
        gather_by_timestep,
        guided_ddim_sample_loop,
        make_diffusion_schedule,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        gather_by_timestep,
        guided_ddim_sample_loop,
        make_diffusion_schedule,
    )


class TwoClassOracle(nn.Module):
    """An exact predictor that makes the effect of guidance easy to inspect."""

    def __init__(self, schedule, targets: torch.Tensor):
        super().__init__()
        self.schedule = schedule
        self.null_class = targets.shape[0]
        self.register_buffer("targets", targets)

    def forward(self, x_t, timesteps, labels):
        target = torch.zeros_like(x_t)
        real = labels != self.null_class
        target[real] = self.targets[labels[real]]
        signal = gather_by_timestep(self.schedule.sqrt_alpha_bars, timesteps, x_t.shape)
        noise_scale = gather_by_timestep(
            self.schedule.sqrt_one_minus_alpha_bars, timesteps, x_t.shape
        )
        return (x_t - signal * target) / noise_scale


def main() -> None:
    torch.manual_seed(6)
    schedule = make_diffusion_schedule(12, beta_start=0.01, beta_end=0.12)
    targets = torch.zeros(2, 1, 8, 8)
    targets[0, :, :, 3:5] = 1.0  # vertical bar
    targets[1, :, 3:5, :] = 1.0  # horizontal bar
    labels = torch.tensor([0, 1])
    initial_noise = torch.randn_like(targets)
    model = TwoClassOracle(schedule, targets)

    print("class 0 target: vertical bar; class 1 target: horizontal bar")
    for scale in (0.0, 1.0, 2.0):
        samples = guided_ddim_sample_loop(
            model,
            schedule,
            shape=tuple(targets.shape),
            class_labels=labels,
            guidance_scale=scale,
            num_steps=5,
            initial_noise=initial_noise,
        )
        target_mae = torch.mean(torch.abs(samples - targets)).item()
        peak = samples.abs().max().item()
        print(f"guidance={scale:.1f} | target MAE={target_mae:.4f} | peak={peak:.2f}")
    print(
        "Scale 0 follows the null target, 1 reaches the class target, and 2 extrapolates."
    )


if __name__ == "__main__":
    main()
