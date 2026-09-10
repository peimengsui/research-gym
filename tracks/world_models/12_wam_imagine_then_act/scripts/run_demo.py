"""Plan in a latent WAM, execute one action, and replan."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import receding_horizon_control  # noqa: E402
from provided import (  # noqa: E402
    CEMConfig,
    ProvidedJointWAM,
    TinyContinuousGrid,
    make_goal_instruction,
)


def main() -> None:
    model = ProvidedJointWAM()
    environment = TinyContinuousGrid(torch.tensor([0.0, 0.0]))
    goal_position = torch.tensor([3, 2])
    goal_ids = make_goal_instruction(goal_position)
    goal_mask = torch.ones_like(goal_ids, dtype=torch.bool)
    config = CEMConfig(
        horizon=3,
        num_iterations=5,
        num_samples=256,
        num_elites=32,
    )

    trace = receding_horizon_control(
        model,
        environment,
        goal_ids,
        goal_mask,
        config,
        max_steps=10,
        generator=torch.Generator().manual_seed(11),
    )
    final_distance = torch.linalg.vector_norm(
        trace.positions[-1] - goal_position.to(torch.float32)
    )

    print(f"goal tokens:       {goal_ids.tolist()} -> reach row 3, column 2")
    print(f"planned horizon:   {config.horizon}")
    print(f"observed positions:{trace.positions.round(decimals=3).tolist()}")
    print(f"executed actions:  {trace.actions.round(decimals=3).tolist()}")
    print(f"planned scores:    {trace.planned_scores.round(decimals=3).tolist()}")
    print(f"final distance:    {final_distance.item():.4f}")
    print(f"reached goal:      {trace.reached_goal}")
    print("Only each plan's first action was executed before observing and replanning.")


if __name__ == "__main__":
    main()
