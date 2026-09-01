import pytest
import torch
import torch.nn.functional as F

from implementation import (
    ActionConditionedJEPA,
    goal_latent_distance,
    rollout_latents,
    select_goal_action_sequence,
    train_action_predictor_step,
)
from provided import (
    apply_grid_actions,
    enumerate_action_sequences,
    make_frozen_jepa_encoder,
    make_grid_transition_batch,
    render_grid_positions,
)


def make_model(seed: int = 0) -> ActionConditionedJEPA:
    torch.manual_seed(seed)
    return ActionConditionedJEPA(
        encoder=make_frozen_jepa_encoder(seed=7),
        action_dim=2,
        hidden_dim=64,
    )


def test_grid_transition_batch_contains_every_state_action_pair() -> None:
    current_images, actions, next_images = make_grid_transition_batch(grid_size=4)

    assert current_images.shape == next_images.shape == (80, 1, 4, 4)
    assert actions.shape == (80, 2)
    assert torch.equal(current_images.sum(dim=(1, 2, 3)), torch.ones(80))
    assert torch.equal(next_images.sum(dim=(1, 2, 3)), torch.ones(80))


def test_encoder_is_carried_forward_and_frozen() -> None:
    model = make_model()

    assert model.num_patches == 4
    assert model.embed_dim == 8
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.predictor.parameters())


def test_forward_returns_patch_latents_and_exact_mse() -> None:
    model = make_model()
    current_images, actions, next_images = make_grid_transition_batch()

    output = model(current_images[:6], actions[:6], next_images[:6])

    assert output.predicted_next_latents.shape == (6, 4, 8)
    assert output.target_next_latents.shape == (6, 4, 8)
    assert output.loss.shape == ()
    assert not output.target_next_latents.requires_grad
    assert torch.allclose(
        output.loss,
        F.mse_loss(output.predicted_next_latents, output.target_next_latents),
    )


def test_loss_trains_predictor_but_not_encoder() -> None:
    model = make_model()
    current_images, actions, next_images = make_grid_transition_batch()

    output = model(current_images, actions, next_images)
    output.loss.backward()

    predictor_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.predictor.parameters()
        if parameter.grad is not None
    )
    assert predictor_gradient > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_action_conditioning_changes_the_predicted_transition() -> None:
    model = make_model()
    image = render_grid_positions(torch.tensor([[1, 1]]))
    current_latents = model.encode_images(image).expand(2, -1, -1)
    actions = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])

    predictions = model.predict_next_latent(current_latents, actions)

    assert not torch.allclose(predictions[0], predictions[1])


def test_training_reduces_one_step_latent_loss() -> None:
    model = make_model(seed=1)
    current_images, actions, next_images = make_grid_transition_batch()
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=0.02)

    with torch.no_grad():
        initial_loss = model(current_images, actions, next_images).loss.item()
    for _ in range(180):
        train_action_predictor_step(
            model,
            current_images,
            actions,
            next_images,
            optimizer,
        )
    with torch.no_grad():
        final_loss = model(current_images, actions, next_images).loss.item()

    assert final_loss < initial_loss * 0.1


class AdditiveLatentModel:
    action_dim = 2
    num_patches = 1
    embed_dim = 2

    def predict_next_latent(
        self,
        current_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return current_latents + actions.unsqueeze(1)


def test_rollout_latents_recursively_applies_actions() -> None:
    model = AdditiveLatentModel()
    initial = torch.zeros(2, 1, 2)
    actions = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[-1.0, 1.0], [2.0, 0.0]],
        ]
    )

    trajectory = rollout_latents(model, initial, actions)

    assert trajectory.shape == (2, 3, 1, 2)
    assert torch.equal(trajectory[:, 0], initial)
    assert torch.equal(trajectory[0, -1], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(trajectory[1, -1], torch.tensor([[1.0, 1.0]]))


def test_goal_latent_distance_scores_each_candidate() -> None:
    predicted = torch.tensor([[[0.0, 0.0]], [[1.0, 2.0]], [[3.0, 3.0]]])
    goal = torch.tensor([[1.0, 2.0]])

    distances = goal_latent_distance(predicted, goal)

    assert distances.shape == (3,)
    assert distances.argmin().item() == 1
    assert distances[1].item() == 0.0


class ExactPlanningModel(AdditiveLatentModel):
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, 0, 0, :].unsqueeze(1)


def test_select_goal_action_sequence_chooses_closest_imagined_state() -> None:
    model = ExactPlanningModel()
    current_image = torch.tensor([[[0.0, 0.0]]])
    goal_image = torch.tensor([[[1.0, 2.0]]])
    candidates = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 2.0]],
            [[-1.0, 0.0], [0.0, -2.0]],
        ]
    )

    best_actions, best_distance = select_goal_action_sequence(
        model,
        current_image,
        goal_image,
        candidates,
    )

    assert torch.equal(best_actions, candidates[1])
    assert best_distance.shape == ()
    assert best_distance.item() == 0.0


def test_enumerated_short_plan_can_reach_a_grid_goal() -> None:
    candidates = enumerate_action_sequences(horizon=3)
    initial_position = torch.tensor([0, 0])
    goal_position = torch.tensor([2, 1])

    final_positions = torch.stack(
        [apply_grid_actions(initial_position, actions)[-1] for actions in candidates]
    )

    assert candidates.shape == (125, 3, 2)
    assert (final_positions == goal_position).all(dim=1).any()


def test_model_rejects_wrong_action_width() -> None:
    model = make_model()
    current_images, _, next_images = make_grid_transition_batch()

    with pytest.raises(ValueError, match="actions"):
        model(current_images[:2], torch.zeros(2, 3), next_images[:2])
