"""Train a tiny masked latent predictor on structured synthetic images."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    TinyJEPA,
    sample_context_target_masks,
    train_jepa_step,
)
from provided import make_structured_images  # noqa: E402


def main() -> None:
    torch.manual_seed(6)
    images = make_structured_images(
        batch_size=32,
        generator=torch.Generator().manual_seed(7),
    )
    model = TinyJEPA(4, 2, 1, 12, 24)
    context_mask, target_mask = sample_context_target_masks(
        batch_size=images.shape[0],
        num_patches=model.num_patches,
        num_targets=2,
        generator=torch.Generator().manual_seed(8),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    with torch.no_grad():
        initial_output = model(images, context_mask, target_mask)
        initial_loss = initial_output.loss.item()
    target_before = [
        parameter.clone() for parameter in model.target_encoder.parameters()
    ]

    for _ in range(40):
        train_jepa_step(
            model,
            images,
            context_mask,
            target_mask,
            optimizer,
            momentum=0.95,
        )

    with torch.no_grad():
        final_output = model(images, context_mask, target_mask)
    target_shift = sum(
        (before - after).abs().sum().item()
        for before, after in zip(
            target_before,
            model.target_encoder.parameters(),
            strict=True,
        )
    )

    print(f"image batch shape:           {tuple(images.shape)}")
    print(f"context patches per image:   {int(context_mask[0].sum())}")
    print(f"target patches per image:    {int(target_mask[0].sum())}")
    print(f"prediction shape:            {tuple(final_output.predictions.shape)}")
    print(f"initial latent loss:         {initial_loss:.4f}")
    print(f"final latent loss:           {final_output.loss.item():.4f}")
    print(f"target encoder EMA shift:    {target_shift:.4f}")
    print("The model predicts target representations; it never reconstructs pixels.")


if __name__ == "__main__":
    main()
