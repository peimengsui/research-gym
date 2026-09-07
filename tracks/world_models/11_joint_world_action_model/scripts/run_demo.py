"""Train action and latent prediction together on a tiny moving-dot world."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    TinyJointWorldActionModel,
    joint_world_action_loss,
    train_joint_wam_step,
)
from provided import (  # noqa: E402
    ActionSpec,
    make_frozen_multimodal_encoders,
    make_toy_wam_batch,
)


def main() -> None:
    torch.manual_seed(11)
    batch = make_toy_wam_batch()
    video_encoder, language_encoder = make_frozen_multimodal_encoders(seed=11)
    model = TinyJointWorldActionModel(
        video_encoder=video_encoder,
        language_encoder=language_encoder,
        action_spec=ActionSpec(
            low=torch.tensor([-1.0, -1.0]),
            high=torch.tensor([1.0, 1.0]),
        ),
        chunk_size=2,
        state_dim=12,
        hidden_dim=48,
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.015,
    )

    action_weight = 4.0
    initial = joint_world_action_loss(model, batch, action_weight=action_weight)
    initial_metrics = (
        initial.total_loss.item(),
        initial.action_loss.item(),
        initial.latent_loss.item(),
    )
    for _ in range(500):
        train_joint_wam_step(model, batch, optimizer, action_weight=action_weight)
    final = joint_world_action_loss(model, batch, action_weight=action_weight)

    model.eval()
    with torch.no_grad():
        output = model(
            batch.current_videos,
            batch.instruction_ids,
            batch.instruction_mask,
            batch.transition_actions,
            batch.next_videos,
        )
        latent_mae = (
            (output.predicted_next_latents - output.target_next_latents).abs().mean()
        )

    example_index = 23
    direction_names = {2: "up", 3: "down", 4: "left", 5: "right"}
    direction_id = batch.instruction_ids[example_index, 1].item()

    print(f"current/next video shape: {tuple(batch.current_videos.shape)}")
    print(f"action chunk shape:       {tuple(batch.action_chunks.shape)}")
    print(f"shared latent shape:      {tuple(output.state_latents.shape)}")
    print(f"initial total loss:       {initial_metrics[0]:.4f}")
    print(f"initial action loss:      {initial_metrics[1]:.4f}")
    print(f"trained action loss:      {final.action_loss.item():.4f}")
    print(f"initial latent loss:      {initial_metrics[2]:.4f}")
    print(f"trained latent loss:      {final.latent_loss.item():.4f}")
    print(f"trained latent MAE:       {latent_mae.item():.4f}")
    print(f"example instruction:      move {direction_names[direction_id]}")
    print(f"expert action chunk:      {batch.action_chunks[example_index].tolist()}")
    print(f"predicted action chunk:   {output.actions[example_index].tolist()}")
    print("The shared state now supports both acting and predicting consequences.")


if __name__ == "__main__":
    main()
