import pytest
import torch
import torch.nn.functional as F

from implementation import (
    TinyJointWorldActionModel,
    joint_world_action_loss,
    train_joint_wam_step,
)
from provided import (
    ActionSpec,
    make_frozen_multimodal_encoders,
    make_toy_wam_batch,
    masked_action_mse,
    normalize_actions,
)


def make_model(seed: int = 0) -> TinyJointWorldActionModel:
    torch.manual_seed(seed)
    video_encoder, language_encoder = make_frozen_multimodal_encoders(seed=11)
    return TinyJointWorldActionModel(
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


def gradient_sum(module: torch.nn.Module) -> float:
    return sum(
        parameter.grad.abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def video_positions(videos: torch.Tensor) -> torch.Tensor:
    flat_indices = videos[:, -1, 0].flatten(start_dim=1).argmax(dim=1)
    return torch.stack((flat_indices // 4, flat_indices % 4), dim=1)


def test_toy_batch_aligns_actions_with_one_step_consequences() -> None:
    batch = make_toy_wam_batch()

    assert batch.current_videos.shape == batch.next_videos.shape == (64, 2, 1, 4, 4)
    assert batch.instruction_ids.shape == batch.instruction_mask.shape == (64, 3)
    assert batch.action_chunks.shape == (64, 2, 2)
    assert batch.transition_actions.shape == (64, 2)
    assert batch.action_validity.shape == (64, 2)
    assert torch.equal(batch.transition_actions, batch.action_chunks[:, 0])

    current_positions = video_positions(batch.current_videos)
    next_positions = video_positions(batch.next_videos)
    expected_positions = (
        current_positions + batch.transition_actions.to(torch.long)
    ).clamp(0, 3)
    assert torch.equal(next_positions, expected_positions)


def test_provided_encoders_are_frozen_but_joint_heads_are_trainable() -> None:
    model = make_model()

    assert all(
        not parameter.requires_grad for parameter in model.video_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in model.language_encoder.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.state_encoder.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.action_head.parameters())
    assert all(
        parameter.requires_grad for parameter in model.dynamics_model.parameters()
    )


def test_forward_returns_bounded_actions_and_detached_latent_targets() -> None:
    model = make_model()
    batch = make_toy_wam_batch()

    output = model(
        batch.current_videos[:5],
        batch.instruction_ids[:5],
        batch.instruction_mask[:5],
        batch.transition_actions[:5],
        batch.next_videos[:5],
    )

    assert output.state_latents.shape == (5, 12)
    assert output.normalized_actions.shape == output.actions.shape == (5, 2, 2)
    assert output.predicted_next_latents.shape == (5, 12)
    assert output.target_next_latents.shape == (5, 12)
    assert (output.normalized_actions >= -1.0).all()
    assert (output.normalized_actions <= 1.0).all()
    assert output.state_latents.requires_grad
    assert output.predicted_next_latents.requires_grad
    assert not output.target_next_latents.requires_grad


def test_dynamics_prediction_changes_with_the_executed_action() -> None:
    model = make_model()
    batch = make_toy_wam_batch()
    state = model.encode_video(batch.current_videos[:1]).expand(2, -1)
    actions = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])

    predictions = model.predict_next_latent(state, actions)

    assert predictions.shape == (2, 12)
    assert not torch.allclose(predictions[0], predictions[1])


def test_joint_loss_matches_unweighted_component_losses() -> None:
    model = make_model()
    batch = make_toy_wam_batch()
    output = model(
        batch.current_videos,
        batch.instruction_ids,
        batch.instruction_mask,
        batch.transition_actions,
        batch.next_videos,
    )
    expected_action = masked_action_mse(
        output.normalized_actions,
        normalize_actions(batch.action_chunks, model.action_spec),
        batch.action_validity,
    )
    expected_latent = F.mse_loss(
        output.predicted_next_latents,
        output.target_next_latents,
    )

    losses = joint_world_action_loss(
        model,
        batch,
        action_weight=2.0,
        latent_weight=0.25,
    )

    assert torch.allclose(losses.action_loss, expected_action)
    assert torch.allclose(losses.latent_loss, expected_latent)
    assert torch.allclose(
        losses.total_loss,
        2.0 * expected_action + 0.25 * expected_latent,
    )


def test_action_objective_uses_shared_state_but_not_dynamics_head() -> None:
    model = make_model()
    batch = make_toy_wam_batch()

    losses = joint_world_action_loss(model, batch, action_weight=1.0, latent_weight=0.0)
    losses.total_loss.backward()

    assert gradient_sum(model.state_encoder) > 0
    assert gradient_sum(model.action_fusion) > 0
    assert gradient_sum(model.action_head) > 0
    assert gradient_sum(model.dynamics_model) == 0


def test_latent_objective_uses_shared_state_but_not_action_head() -> None:
    model = make_model()
    batch = make_toy_wam_batch()

    losses = joint_world_action_loss(model, batch, action_weight=0.0, latent_weight=1.0)
    losses.total_loss.backward()

    assert gradient_sum(model.state_encoder) > 0
    assert gradient_sum(model.dynamics_model) > 0
    assert gradient_sum(model.action_fusion) == 0
    assert gradient_sum(model.action_head) == 0


def test_joint_training_reduces_both_objectives() -> None:
    model = make_model(seed=11)
    batch = make_toy_wam_batch()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.015,
    )

    initial = joint_world_action_loss(model, batch, action_weight=4.0)
    initial_action = initial.action_loss.item()
    initial_latent = initial.latent_loss.item()
    for _ in range(500):
        train_joint_wam_step(model, batch, optimizer, action_weight=4.0)
    final = joint_world_action_loss(model, batch, action_weight=4.0)

    assert final.action_loss.item() < initial_action * 0.1
    assert final.latent_loss.item() < initial_latent * 0.1


def test_batch_rejects_transition_action_that_does_not_match_chunk() -> None:
    model = make_model()
    batch = make_toy_wam_batch()
    batch.transition_actions = batch.transition_actions.clone()
    batch.transition_actions[0] = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="first action"):
        joint_world_action_loss(model, batch)
