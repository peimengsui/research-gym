import pytest
import torch
import torch.nn.functional as F

from implementation import (
    TinyWorldModel,
    plan_with_cem,
    sample_action_sequences,
    select_elites,
    trajectory_reward,
    update_action_distribution,
)


def make_observation_batch(
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return observations[:, :-1, :], actions, observations[:, 1:, :]


def make_toy_sequences(
    batch_size: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.zeros(batch_size, horizon + 1, 2)
    actions = torch.randn(batch_size, horizon, 2) * 0.5
    observations[:, 0, :] = torch.randn(batch_size, 2) * 0.2
    for step in range(horizon):
        observations[:, step + 1, :] = (
            0.75 * observations[:, step, :] + actions[:, step, :]
        )
    return observations, actions


def world_model_loss(
    reconstruction: torch.Tensor,
    current_observations: torch.Tensor,
    predicted_next_observations: torch.Tensor,
    next_observations: torch.Tensor,
    predicted_next_latents: torch.Tensor,
    target_next_latents: torch.Tensor,
    latent_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(reconstruction, current_observations)
    prediction_loss = F.mse_loss(predicted_next_observations, next_observations)
    latent_loss = F.mse_loss(predicted_next_latents, target_next_latents.detach())
    total = reconstruction_loss + prediction_loss + latent_weight * latent_loss
    return total, reconstruction_loss, prediction_loss, latent_loss


def train_tiny_world_model(
    seed: int = 0,
    steps: int = 160,
) -> TinyWorldModel:
    torch.manual_seed(seed)
    model = TinyWorldModel(obs_dim=2, action_dim=2, latent_dim=4, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    observations, actions = make_toy_sequences(batch_size=64, horizon=5)
    current_obs, current_actions, next_obs = make_observation_batch(
        observations,
        actions,
    )

    model.train()
    for _ in range(steps):
        reconstruction = model.decode(model.encode(current_obs))
        predicted_next_z = model.predict_next_latent(
            model.encode(current_obs),
            current_actions,
        )
        predicted_next_obs = model.decode(predicted_next_z)
        target_next_z = model.encode(next_obs)
        loss, _, _, _ = world_model_loss(
            reconstruction,
            current_obs,
            predicted_next_obs,
            next_obs,
            predicted_next_z,
            target_next_z,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def test_trajectory_reward_prefers_closer_final_states() -> None:
    predicted_observations = torch.tensor(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.5, 0.5], [0.9, 0.9]],
        ]
    )
    target = torch.tensor([1.0, 1.0])

    rewards = trajectory_reward(predicted_observations, target)

    assert rewards.shape == (2,)
    assert rewards[1] > rewards[0]


def test_sample_action_sequences_returns_expected_shape() -> None:
    mean = torch.zeros(4, 2)
    std = torch.full((4, 2), 0.5)
    generator = torch.Generator().manual_seed(0)

    action_sequences = sample_action_sequences(
        mean, std, num_samples=8, generator=generator
    )

    assert action_sequences.shape == (8, 4, 2)
    assert torch.isfinite(action_sequences).all()


def test_select_elites_returns_highest_reward_sequences() -> None:
    action_sequences = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[3.0, 0.0], [4.0, 0.0]],
            [[5.0, 0.0], [6.0, 0.0]],
        ]
    )
    rewards = torch.tensor([-3.0, -1.0, -2.0])

    elite_actions = select_elites(action_sequences, rewards, num_elites=2)

    assert elite_actions.shape == (2, 2, 2)
    assert torch.equal(elite_actions[0], action_sequences[1])
    assert torch.equal(elite_actions[1], action_sequences[2])


def test_update_action_distribution_fits_elites_and_clamps_std() -> None:
    elite_actions = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[2.0, 0.0], [3.0, 1.0]],
        ]
    )

    mean, std = update_action_distribution(elite_actions, min_std=0.25)

    assert mean.shape == (2, 2)
    assert std.shape == (2, 2)
    assert torch.allclose(mean, torch.tensor([[1.0, 0.0], [2.0, 1.0]]))
    assert (std >= 0.25).all()


@pytest.mark.parametrize(
    ("initial_observation", "target", "horizon", "action_dim"),
    [
        (torch.zeros(2), torch.ones(2), 0, 2),
        (torch.zeros(2), torch.ones(2), 4, 0),
        (torch.zeros(2, 1), torch.ones(2), 4, 2),
        (torch.zeros(2), torch.ones(3), 4, 2),
    ],
)
def test_plan_with_cem_rejects_invalid_shapes(
    initial_observation: torch.Tensor,
    target: torch.Tensor,
    horizon: int,
    action_dim: int,
) -> None:
    model = TinyWorldModel(obs_dim=2, action_dim=2, latent_dim=4, hidden_dim=8)

    with pytest.raises(ValueError):
        plan_with_cem(
            model,
            initial_observation,
            target,
            horizon=horizon,
            action_dim=action_dim,
            num_iterations=2,
            num_samples=4,
            num_elites=2,
            initial_std=1.0,
            min_std=0.1,
        )


def test_plan_with_cem_improves_over_zero_actions() -> None:
    torch.manual_seed(1)
    model = train_tiny_world_model(seed=1)
    initial_observation = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, 1.0])
    horizon = 5
    zero_actions = torch.zeros(horizon, 2)

    with torch.no_grad():
        zero_rollout, _ = model.rollout(
            initial_observation.unsqueeze(0),
            zero_actions.unsqueeze(0),
        )
        zero_reward = trajectory_reward(zero_rollout, target)

    planned_actions, planned_reward = plan_with_cem(
        model,
        initial_observation,
        target,
        horizon=horizon,
        action_dim=2,
        num_iterations=6,
        num_samples=128,
        num_elites=16,
        initial_std=1.0,
        min_std=0.05,
        generator=torch.Generator().manual_seed(2),
    )

    assert planned_actions.shape == (horizon, 2)
    assert planned_reward.shape == ()
    assert planned_reward > zero_reward


def test_plan_with_cem_reaches_target_more_closely_than_random_search() -> None:
    torch.manual_seed(3)
    model = train_tiny_world_model(seed=3)
    initial_observation = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, 1.0])
    horizon = 6
    generator = torch.Generator().manual_seed(4)

    random_actions = torch.randn(horizon, 2, generator=generator)
    with torch.no_grad():
        random_rollout, _ = model.rollout(
            initial_observation.unsqueeze(0),
            random_actions.unsqueeze(0),
        )
        random_distance = ((random_rollout[:, -1, :] - target) ** 2).sum()

    planned_actions, _ = plan_with_cem(
        model,
        initial_observation,
        target,
        horizon=horizon,
        action_dim=2,
        num_iterations=8,
        num_samples=256,
        num_elites=32,
        initial_std=1.0,
        min_std=0.05,
        generator=torch.Generator().manual_seed(5),
    )

    with torch.no_grad():
        planned_rollout, _ = model.rollout(
            initial_observation.unsqueeze(0),
            planned_actions.unsqueeze(0),
        )
        planned_distance = ((planned_rollout[:, -1, :] - target) ** 2).sum()

    assert planned_distance < random_distance * 0.5
