"""Train tiny actor and value networks entirely from latent imagination."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import train_actor_critic_step  # noqa: E402
from provided import (  # noqa: E402
    ConstantContinuationPredictor,
    GoalRewardPredictor,
    ProvidedActionConditionedWorldModel,
    TinyActor,
    TinyValuePredictor,
    imagine_for_actor,
)


def evaluate(
    world_model: ProvidedActionConditionedWorldModel,
    actor: TinyActor,
    reward: GoalRewardPredictor,
    continuation: ConstantContinuationPredictor,
    value: TinyValuePredictor,
    initial_latents: torch.Tensor,
) -> tuple[float, float, torch.Tensor]:
    with torch.no_grad():
        imagined = imagine_for_actor(
            world_model,
            actor,
            reward,
            continuation,
            value,
            initial_latents,
            horizon=4,
            discount=0.9,
            lambda_=0.8,
        )
        goal = torch.tensor([[2.0, 1.0]])
        final_distance = ((imagined.states[:, -1] - goal) ** 2).mean().item()
        mean_return = imagined.lambda_returns.mean().item()
    return final_distance, mean_return, imagined.states


def main() -> None:
    torch.manual_seed(9)
    initial_latents = torch.tensor(
        [
            [[-2.0, -1.0]],
            [[-1.0, 0.0]],
            [[0.0, -1.0]],
            [[0.0, 0.0]],
        ]
    )
    world_model = ProvidedActionConditionedWorldModel()
    actor = TinyActor(1, 2, 2, 32)
    reward = GoalRewardPredictor(torch.tensor([[2.0, 1.0]]))
    continuation = ConstantContinuationPredictor(1.0)
    value = TinyValuePredictor(1, 2, 32)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.015)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=0.015)

    initial_distance, initial_return, initial_states = evaluate(
        world_model,
        actor,
        reward,
        continuation,
        value,
        initial_latents,
    )
    metrics = None
    for _ in range(120):
        metrics = train_actor_critic_step(
            world_model,
            actor,
            reward,
            continuation,
            value,
            initial_latents,
            actor_optimizer,
            value_optimizer,
            horizon=4,
            discount=0.9,
            lambda_=0.8,
        )
    final_distance, final_return, final_states = evaluate(
        world_model,
        actor,
        reward,
        continuation,
        value,
        initial_latents,
    )
    assert metrics is not None

    print(f"initial final-goal MSE:     {initial_distance:.4f}")
    print(f"trained final-goal MSE:     {final_distance:.4f}")
    print(f"initial mean return:        {initial_return:.4f}")
    print(f"trained mean return:        {final_return:.4f}")
    print(f"last actor loss:            {metrics.actor_loss:.4f}")
    print(f"last value loss:            {metrics.value_loss:.4f}")
    print(f"first rollout before:       {initial_states[0, :, 0].tolist()}")
    print(f"first rollout after:        {final_states[0, :, 0].tolist()}")
    print("Actor gradients crossed imagined dynamics; critic targets were detached.")


if __name__ == "__main__":
    main()
