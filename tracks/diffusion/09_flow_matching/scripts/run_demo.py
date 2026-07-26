"""Compare ODE solvers and trace a noise-to-image flow."""

import math
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import TargetVelocityField, ode_sample
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import TargetVelocityField, ode_sample


class ExponentialField(nn.Module):
    """A known curved ODE: dx/dt = x, whose solution from x(0)=1 is e^t."""

    def forward(self, x, times):
        return x


def main() -> None:
    initial = torch.ones(1, 1)
    euler = ode_sample(
        ExponentialField(), (1, 1), 4, solver="euler", initial_noise=initial
    )
    midpoint = ode_sample(
        ExponentialField(), (1, 1), 4, solver="midpoint", initial_noise=initial
    )
    exact = math.e

    torch.manual_seed(9)
    target = torch.full((1, 1, 8, 8), -1.0)
    target[:, :, 2:6, 2:6] = 1.0
    initial_noise = torch.randn_like(target)
    sample, trajectory = ode_sample(
        TargetVelocityField(target),
        tuple(target.shape),
        num_steps=5,
        solver="euler",
        initial_noise=initial_noise,
        return_all=True,
    )

    print("Known ODE dx/dt = x from t=0 to t=1:")
    print(f"  exact value:       {exact:.6f}")
    print(f"  Euler, 4 steps:    {euler.item():.6f}")
    print(f"  midpoint, 4 steps: {midpoint.item():.6f}")
    print()
    print(f"flow trajectory states: {len(trajectory)}")
    print(f"initial noise mean:     {initial_noise.mean().item():.4f}")
    print(
        f"target recovery MAE:    {torch.mean(torch.abs(sample - target)).item():.6f}"
    )
    print("Flow matching generates by integrating a velocity field from t=0 to t=1.")


if __name__ == "__main__":
    main()
