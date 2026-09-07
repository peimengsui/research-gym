import pytest
import torch

from implementation import (
    TinyVLAPolicy,
    behavior_cloning_loss,
    denormalize_actions,
    masked_action_mse,
    normalize_actions,
    train_vla_step,
)
from provided import ActionSpec, make_tiny_multimodal_encoders, make_toy_vla_batch


def make_policy(seed: int = 0) -> TinyVLAPolicy:
    torch.manual_seed(seed)
    video_encoder, language_encoder = make_tiny_multimodal_encoders()
    action_spec = ActionSpec(
        low=torch.tensor([-1.0, -1.0]),
        high=torch.tensor([1.0, 1.0]),
    )
    return TinyVLAPolicy(
        video_encoder,
        language_encoder,
        action_spec,
        chunk_size=2,
        hidden_dim=32,
    )


def test_toy_batch_aligns_modalities_and_action_chunks() -> None:
    batch = make_toy_vla_batch()

    assert batch.videos.shape == (64, 2, 1, 4, 4)
    assert batch.instruction_ids.shape == (64, 3)
    assert batch.instruction_mask.shape == (64, 3)
    assert batch.actions.shape == (64, 2, 2)
    assert batch.action_validity.shape == (64, 2)
    assert batch.instruction_mask.dtype == torch.bool
    assert batch.action_validity.all()


def test_action_normalization_maps_bounds_and_round_trips() -> None:
    action_spec = ActionSpec(
        low=torch.tensor([-2.0, 10.0]),
        high=torch.tensor([2.0, 30.0]),
    )
    actions = torch.tensor([[[-2.0, 10.0], [0.0, 20.0], [2.0, 30.0]]])

    normalized = normalize_actions(actions, action_spec)

    expected = torch.tensor([[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]])
    assert torch.allclose(normalized, expected)
    assert torch.allclose(denormalize_actions(normalized, action_spec), actions)


def test_action_normalization_rejects_values_outside_bounds() -> None:
    action_spec = ActionSpec(torch.tensor([-1.0]), torch.tensor([1.0]))

    with pytest.raises(ValueError, match="within the action specification"):
        normalize_actions(torch.tensor([[[1.1]]]), action_spec)


def test_policy_returns_bounded_action_chunks() -> None:
    model = make_policy()
    batch = make_toy_vla_batch()

    output = model(
        batch.videos[:5],
        batch.instruction_ids[:5],
        batch.instruction_mask[:5],
    )

    assert output.normalized_actions.shape == (5, 2, 2)
    assert output.actions.shape == (5, 2, 2)
    assert (output.normalized_actions >= -1.0).all()
    assert (output.normalized_actions <= 1.0).all()
    assert torch.allclose(output.actions, output.normalized_actions)


def test_masked_action_mse_ignores_invalid_chunk_steps() -> None:
    predictions = torch.zeros(1, 2, 2)
    targets = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    validity = torch.tensor([[True, False]])

    loss = masked_action_mse(predictions, targets, validity)

    assert torch.allclose(loss, torch.tensor(2.5))


def test_behavior_cloning_loss_uses_normalized_expert_actions() -> None:
    model = make_policy()
    batch = make_toy_vla_batch()

    output = model(batch.videos, batch.instruction_ids, batch.instruction_mask)
    expected = masked_action_mse(
        output.normalized_actions,
        normalize_actions(batch.actions, model.action_spec),
        batch.action_validity,
    )

    assert torch.allclose(behavior_cloning_loss(model, batch), expected)


def test_behavior_cloning_gradients_reach_both_modalities_and_action_head() -> None:
    model = make_policy()
    batch = make_toy_vla_batch()

    behavior_cloning_loss(model, batch).backward()

    video_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.video_encoder.parameters()
        if parameter.grad is not None
    )
    language_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.language_encoder.parameters()
        if parameter.grad is not None
    )
    action_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.action_head.parameters()
        if parameter.grad is not None
    )
    assert video_gradient > 0
    assert language_gradient > 0
    assert action_gradient > 0


def test_training_overfits_the_toy_expert_demonstrations() -> None:
    model = make_policy(seed=10)
    batch = make_toy_vla_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    initial_loss = behavior_cloning_loss(model, batch).item()
    for _ in range(250):
        train_vla_step(model, batch, optimizer)
    final_loss = behavior_cloning_loss(model, batch).item()

    assert final_loss < initial_loss * 0.05
    assert final_loss < 0.02


def test_policy_rejects_an_instruction_with_no_valid_tokens() -> None:
    model = make_policy()
    batch = make_toy_vla_batch()
    invalid_mask = batch.instruction_mask[:2].clone()
    invalid_mask[0] = False

    with pytest.raises(ValueError, match="valid token"):
        model(batch.videos[:2], batch.instruction_ids[:2], invalid_mask)
