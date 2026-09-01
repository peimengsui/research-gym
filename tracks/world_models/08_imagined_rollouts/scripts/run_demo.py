"""Imagine a goal-directed latent trajectory and compare lambda returns."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import imagine_and_compute_returns, lambda_returns  # noqa: E402
from provided import (  # noqa: E402
    GoalContinuationPredictor,
    GoalDirectedPolicy,
    GoalRewardPredictor,
    GoalValuePredictor,
    ProvidedActionConditionedWorldModel,
)


def main() -> None:
    goal_latents = torch.tensor([[2.0, 1.0]])
    initial_latents = torch.tensor([[[0.0, 0.0]]])
    world_model = ProvidedActionConditionedWorldModel()
    policy = GoalDirectedPolicy(goal_latents)
    reward_predictor = GoalRewardPredictor(goal_latents)
    continuation_predictor = GoalContinuationPredictor(goal_latents)
    value_predictor = GoalValuePredictor(goal_latents)

    imagined = imagine_and_compute_returns(
        world_model,
        policy,
        reward_predictor,
        continuation_predictor,
        value_predictor,
        initial_latents,
        horizon=4,
        discount=0.9,
        lambda_=0.5,
    )
    one_step_returns = lambda_returns(
        imagined.rewards,
        imagined.discounts,
        imagined.values,
        lambda_=0.0,
    )
    monte_carlo_returns = lambda_returns(
        imagined.rewards,
        imagined.discounts,
        imagined.values,
        lambda_=1.0,
    )

    print(f"state trajectory shape:     {tuple(imagined.states.shape)}")
    print(f"action trajectory shape:    {tuple(imagined.actions.shape)}")
    print(f"imagined states:            {imagined.states[0, :, 0].tolist()}")
    print(f"imagined actions:           {imagined.actions[0].tolist()}")
    print(f"predicted rewards:          {imagined.rewards[0].tolist()}")
    print(f"continuation probabilities: {imagined.continuations[0].tolist()}")
    print(f"effective discounts:        {imagined.discounts[0].tolist()}")
    print(f"predicted values:           {imagined.values[0].tolist()}")
    print(f"lambda=0 returns:           {one_step_returns[0].tolist()}")
    print(f"lambda=0.5 returns:         {imagined.lambda_returns[0].tolist()}")
    print(f"lambda=1 returns:           {monte_carlo_returns[0].tolist()}")
    print("The terminal continuation of zero prevents bootstrapping past the goal.")


if __name__ == "__main__":
    main()
