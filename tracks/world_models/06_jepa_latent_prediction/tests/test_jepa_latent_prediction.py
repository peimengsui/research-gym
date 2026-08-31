from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from implementation import (
    TinyJEPA,
    sample_context_target_masks,
    select_masked_tokens,
    train_jepa_step,
    update_target_encoder,
)
from provided import make_structured_images


def make_model() -> TinyJEPA:
    torch.manual_seed(0)
    return TinyJEPA(
        image_size=4,
        patch_size=2,
        in_channels=1,
        embed_dim=8,
        predictor_hidden_dim=16,
    )


def make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = make_structured_images(
        batch_size=6,
        generator=torch.Generator().manual_seed(1),
    )
    context_mask, target_mask = sample_context_target_masks(
        batch_size=6,
        num_patches=4,
        num_targets=2,
        generator=torch.Generator().manual_seed(2),
    )
    return images, context_mask, target_mask


def test_sample_masks_have_expected_counts_and_are_disjoint() -> None:
    context_mask, target_mask = sample_context_target_masks(
        batch_size=5,
        num_patches=7,
        num_targets=3,
        generator=torch.Generator().manual_seed(3),
    )

    assert context_mask.shape == target_mask.shape == (5, 7)
    assert context_mask.dtype == target_mask.dtype == torch.bool
    assert torch.equal(target_mask.sum(dim=1), torch.full((5,), 3))
    assert torch.equal(context_mask.sum(dim=1), torch.full((5,), 4))
    assert not (context_mask & target_mask).any()
    assert (context_mask | target_mask).all()


def test_sample_masks_are_reproducible_with_generator() -> None:
    first = sample_context_target_masks(
        3, 6, 2, generator=torch.Generator().manual_seed(4)
    )
    second = sample_context_target_masks(
        3, 6, 2, generator=torch.Generator().manual_seed(4)
    )

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_select_masked_tokens_preserves_patch_order() -> None:
    tokens = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[False, True, False, True], [True, False, True, False]])

    selected = select_masked_tokens(tokens, mask)

    assert selected.shape == (2, 2, 3)
    assert torch.equal(selected[0], tokens[0, [1, 3]])
    assert torch.equal(selected[1], tokens[1, [0, 2]])


def test_select_masked_tokens_rejects_ragged_rows() -> None:
    tokens = torch.zeros(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    with pytest.raises(ValueError, match="same number"):
        select_masked_tokens(tokens, mask)


def test_target_encoder_starts_equal_and_frozen() -> None:
    model = make_model()

    for online, target in zip(
        model.online_encoder.parameters(),
        model.target_encoder.parameters(),
        strict=True,
    ):
        assert torch.equal(online, target)
        assert not target.requires_grad
    assert model.online_encoder is not model.target_encoder


def test_forward_returns_target_patch_latents_and_mse() -> None:
    model = make_model()
    images, context_mask, target_mask = make_batch()

    output = model(images, context_mask, target_mask)

    assert output.predictions.shape == (6, 2, 8)
    assert output.target_latents.shape == (6, 2, 8)
    assert output.loss.shape == ()
    assert torch.isfinite(output.loss)
    assert not output.target_latents.requires_grad
    assert torch.allclose(
        output.loss,
        F.mse_loss(output.predictions, output.target_latents),
    )


def test_prediction_loss_trains_online_branch_but_not_target_branch() -> None:
    model = make_model()
    images, context_mask, target_mask = make_batch()

    output = model(images, context_mask, target_mask)
    output.loss.backward()

    online_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.online_encoder.parameters()
        if parameter.grad is not None
    )
    predictor_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.predictor.parameters()
        if parameter.grad is not None
    )
    assert online_gradient > 0
    assert predictor_gradient > 0
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )


def test_update_target_encoder_applies_exact_ema() -> None:
    model = make_model()
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.fill_(2.0)
        for parameter in model.target_encoder.parameters():
            parameter.zero_()

    update_target_encoder(model, momentum=0.75)

    for parameter in model.target_encoder.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.5))
        assert not parameter.requires_grad


def test_train_step_updates_online_parameters_then_target_ema() -> None:
    model = make_model()
    images, context_mask, target_mask = make_batch()
    online_before = deepcopy(tuple(model.online_encoder.parameters()))
    target_before = deepcopy(tuple(model.target_encoder.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    loss = train_jepa_step(
        model,
        images,
        context_mask,
        target_mask,
        optimizer,
        momentum=0.9,
    )

    assert isinstance(loss, float)
    assert loss > 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            online_before, model.online_encoder.parameters(), strict=True
        )
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            target_before, model.target_encoder.parameters(), strict=True
        )
    )
    assert all(
        not parameter.requires_grad for parameter in model.target_encoder.parameters()
    )


def test_forward_rejects_overlapping_context_and_target_masks() -> None:
    model = make_model()
    images, context_mask, target_mask = make_batch()
    context_mask[:, 0] = True
    target_mask[:, 0] = True

    with pytest.raises(ValueError, match="disjoint"):
        model(images, context_mask, target_mask)
