"""Train a tiny action-conditioned latent model and plan to an image goal."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    ActionConditionedJEPA,
    select_goal_action_sequence,
    train_action_predictor_step,
)
from provided import (  # noqa: E402
    apply_grid_actions,
    enumerate_action_sequences,
    make_frozen_jepa_encoder,
    make_grid_transition_batch,
    render_grid_positions,
)


def main() -> None:
    torch.manual_seed(7)
    current_images, actions, next_images = make_grid_transition_batch()
    model = ActionConditionedJEPA(
        encoder=make_frozen_jepa_encoder(seed=6),
        action_dim=2,
        hidden_dim=64,
    )
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=0.02)

    with torch.no_grad():
        initial_loss = model(current_images, actions, next_images).loss.item()
    for _ in range(220):
        train_action_predictor_step(
            model,
            current_images,
            actions,
            next_images,
            optimizer,
        )
    with torch.no_grad():
        final_loss = model(current_images, actions, next_images).loss.item()

    initial_position = torch.tensor([0, 0])
    goal_position = torch.tensor([2, 1])
    current_image = render_grid_positions(initial_position.unsqueeze(0))[0]
    goal_image = render_grid_positions(goal_position.unsqueeze(0))[0]
    candidates = enumerate_action_sequences(horizon=3)
    best_actions, best_distance = select_goal_action_sequence(
        model,
        current_image,
        goal_image,
        candidates,
    )
    realized_positions = apply_grid_actions(initial_position, best_actions)

    print(f"transition batch shape:      {tuple(current_images.shape)}")
    print(f"patch latent shape:          {(model.num_patches, model.embed_dim)}")
    print(f"initial one-step loss:       {initial_loss:.4f}")
    print(f"final one-step loss:         {final_loss:.4f}")
    print(f"candidate action sequences: {candidates.shape[0]}")
    print(f"selected actions [dy, dx]:  {best_actions.tolist()}")
    print(f"imagined goal distance:      {best_distance.item():.4f}")
    print(f"realized grid positions:     {realized_positions.tolist()}")
    print(
        f"reached goal:                {torch.equal(realized_positions[-1], goal_position)}"
    )
    print("Planning used latent predictions; no image decoder was involved.")


if __name__ == "__main__":
    main()
