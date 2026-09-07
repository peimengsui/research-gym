"""Overfit a tiny VLA policy on synthetic grid demonstrations."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    TinyVLAPolicy,
    behavior_cloning_loss,
    train_vla_step,
)
from provided import (  # noqa: E402
    ActionSpec,
    make_tiny_multimodal_encoders,
    make_toy_vla_batch,
)


def main() -> None:
    torch.manual_seed(10)
    batch = make_toy_vla_batch()
    video_encoder, language_encoder = make_tiny_multimodal_encoders()
    action_spec = ActionSpec(
        low=torch.tensor([-1.0, -1.0]),
        high=torch.tensor([1.0, 1.0]),
    )
    model = TinyVLAPolicy(
        video_encoder,
        language_encoder,
        action_spec,
        chunk_size=2,
        hidden_dim=32,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    initial_loss = behavior_cloning_loss(model, batch).item()
    for _ in range(250):
        train_vla_step(model, batch, optimizer)

    model.eval()
    with torch.no_grad():
        output = model(
            batch.videos,
            batch.instruction_ids,
            batch.instruction_mask,
        )
        final_loss = behavior_cloning_loss(model, batch).item()
        valid = batch.action_validity.unsqueeze(-1).expand_as(batch.actions)
        action_mae = (output.actions - batch.actions).abs().masked_select(valid).mean()

    direction_names = {2: "up", 3: "down", 4: "left", 5: "right"}
    example_index = 23
    direction_id = batch.instruction_ids[example_index, 1].item()

    print(f"video batch shape:       {tuple(batch.videos.shape)}")
    print(f"instruction shape:       {tuple(batch.instruction_ids.shape)}")
    print(f"action chunk shape:      {tuple(batch.actions.shape)}")
    print(f"initial cloning loss:    {initial_loss:.4f}")
    print(f"trained cloning loss:    {final_loss:.4f}")
    print(f"trained action MAE:      {action_mae.item():.4f}")
    print(f"example instruction:     move {direction_names[direction_id]}")
    print(f"expert action chunk:     {batch.actions[example_index].tolist()}")
    print(f"predicted action chunk:  {output.actions[example_index].tolist()}")
    print("This policy imitates actions directly; it does not predict future states.")


if __name__ == "__main__":
    main()
