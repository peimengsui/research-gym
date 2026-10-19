"""Overfit a tiny denoiser and show its iterative unmasking trajectory."""

import torch

from implementation import (
    TinyDiscreteDenoiser,
    discrete_denoising_loss,
    iterative_unmask,
)
from provided import eligible_token_mask, make_toy_token_batch


TOKEN_NAMES = {
    0: "[PAD]",
    1: "[BOS]",
    2: "[EOS]",
    3: "red",
    4: "small",
    5: "bird",
    6: "green",
    7: "bright",
    8: "quiet",
    9: "field",
    10: "[MASK]",
}


def render(token_ids: torch.Tensor) -> str:
    return " ".join(TOKEN_NAMES[int(token_id)] for token_id in token_ids)


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    data = make_toy_token_batch()
    num_timesteps = 6
    model = TinyDiscreteDenoiser(
        vocabulary_size=data.vocabulary_size,
        max_length=data.input_ids.shape[1],
        num_timesteps=num_timesteps,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(11)

    model.train()
    for step in range(120):
        timesteps = torch.randint(
            1,
            num_timesteps,
            (data.input_ids.shape[0],),
            generator=generator,
        )
        loss = discrete_denoising_loss(
            model,
            data.input_ids,
            data.attention_mask,
            data.special_token_mask,
            data.mask_token_id,
            timesteps=timesteps,
            generator=generator,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in {0, 39, 79, 119}:
            print(f"training step {step + 1:>3}: loss={loss.item():.4f}")

    editable = eligible_token_mask(data.attention_mask, data.special_token_mask)
    initial_ids = torch.where(
        editable,
        torch.full_like(data.input_ids, data.mask_token_id),
        data.input_ids,
    )
    model.eval()
    final_ids, trajectory = iterative_unmask(
        model,
        initial_ids[:1],
        data.attention_mask[:1],
        data.special_token_mask[:1],
        data.mask_token_id,
        num_timesteps,
        forbidden_token_ids=(0, 1, 2, data.mask_token_id),
    )

    print("\nconfidence-based reverse trajectory:")
    for index, state in enumerate(trajectory[:, 0]):
        print(f"  state {index}: {render(state)}")
    print(f"\nfinal:  {render(final_ids[0])}")
    print(f"target: {render(data.input_ids[0])}")


if __name__ == "__main__":
    main()
