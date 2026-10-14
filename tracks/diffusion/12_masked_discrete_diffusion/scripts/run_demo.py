"""Print one absorbing masked-token diffusion trajectory."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import coupled_mask_trajectory, linear_mask_schedule  # noqa: E402
from provided import make_toy_token_batch  # noqa: E402


TOKEN_TEXT = {
    0: "<pad>",
    1: "<bos>",
    2: "<eos>",
    3: "small",
    4: "red",
    5: "fox",
    6: "blue",
    7: "bird",
    8: "green",
    9: "tree",
    10: "[MASK]",
}


def render_sequence(token_ids: torch.Tensor) -> str:
    return " ".join(TOKEN_TEXT[token_id.item()] for token_id in token_ids)


def main() -> None:
    tokens = make_toy_token_batch()
    schedule = linear_mask_schedule(5)
    trajectory = coupled_mask_trajectory(
        tokens.input_ids,
        schedule,
        tokens.mask_token_id,
        tokens.attention_mask,
        tokens.special_token_mask,
        generator=torch.Generator().manual_seed(12),
    )

    print("First sequence along one coupled forward trajectory:")
    for timestep, state in enumerate(trajectory):
        masked = (state[0] == tokens.mask_token_id).sum().item()
        print(
            f"t={timestep}  p={schedule[timestep]:.2f}  masked={masked}:  "
            f"{render_sequence(state[0])}"
        )
    print("\nBOS, EOS, and padding remain fixed; content masks never reverse.")


if __name__ == "__main__":
    main()
