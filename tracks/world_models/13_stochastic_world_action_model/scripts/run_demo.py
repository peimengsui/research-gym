"""Compare an averaged strategy with coherent stochastic WAM samples."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    sample_coherent_hypotheses,
    score_strategy_hypotheses,
    select_best_strategy,
)
from provided import (  # noqa: E402
    ProvidedStochasticWAM,
    make_branching_problem,
    mixture_mean_strategy,
    strategy_hits_obstacle,
    unpack_strategy_vectors,
)


def main() -> None:
    model = ProvidedStochasticWAM()
    problem = make_branching_problem()
    current, goal, prediction = model(
        problem.current_videos,
        problem.goal_instruction_ids,
        problem.goal_instruction_mask,
    )

    mean_vector = mixture_mean_strategy(prediction)
    mean_actions, mean_futures = unpack_strategy_vectors(
        mean_vector,
        model.horizon,
        model.action_dim,
        model.state_dim,
    )
    mean_collides = strategy_hits_obstacle(
        mean_futures,
        problem.obstacle_latent,
        collision_radius=0.3,
    )

    hypotheses = sample_coherent_hypotheses(
        prediction,
        num_samples=16,
        horizon=model.horizon,
        action_dim=model.action_dim,
        state_dim=model.state_dim,
        generator=torch.Generator().manual_seed(13),
    )
    scores = score_strategy_hypotheses(
        hypotheses,
        current,
        goal,
        problem.obstacle_latent,
    )
    selected = select_best_strategy(hypotheses, scores)
    sampled_collisions = strategy_hits_obstacle(
        hypotheses.future_latents,
        problem.obstacle_latent,
        collision_radius=0.3,
    )

    route_name = "left" if selected.component_ids[0].item() == 0 else "right"
    print(f"mixture probabilities: {prediction.logits.softmax(dim=1)[0].tolist()}")
    print(f"mean action chunk:     {mean_actions[0].tolist()}")
    print(f"mean future path:      {mean_futures[0].tolist()}")
    print(f"mean hits obstacle:    {mean_collides[0].item()}")
    print(f"sampled component ids: {hypotheses.component_ids[0].tolist()}")
    print(f"sample collisions:     {sampled_collisions[0].sum().item()} / 16")
    print(f"selected route:        {route_name}")
    print(
        f"selected actions:      {selected.action_chunks[0].round(decimals=3).tolist()}"
    )
    print(
        f"selected future path:  {selected.future_latents[0].round(decimals=3).tolist()}"
    )
    print("Averaging modes goes through the obstacle; sampling preserves a route.")


if __name__ == "__main__":
    main()
