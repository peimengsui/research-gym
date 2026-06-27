import pytest
import torch

from implementation import TinyWorldModel, make_observation_batch, world_model_loss


def make_toy_sequences(
    batch_size: int = 16,
    horizon: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.zeros(batch_size, horizon + 1, 2)
    actions = torch.randn(batch_size, horizon, 2) * 0.5
    observations[:, 0, :] = torch.randn(batch_size, 2) * 0.2
    for step in range(horizon):
        observations[:, step + 1, :] = (
            0.75 * observations[:, step, :] + actions[:, step, :]
        )
    return observations, actions


def test_make_observation_batch_slices_aligned_sequences() -> None:
    observations = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            [[10.0, 11.0], [11.0, 12.0], [12.0, 13.0]],
        ]
    )
    actions = torch.tensor(
        [
            [[0.5], [1.5]],
            [[2.5], [3.5]],
        ]
    )

    current_obs, current_actions, next_obs = make_observation_batch(
        observations,
        actions,
    )

    assert current_obs.shape == (2, 2, 2)
    assert current_actions.shape == (2, 2, 1)
    assert next_obs.shape == (2, 2, 2)
    assert torch.equal(current_obs[0, 0], torch.tensor([0.0, 1.0]))
    assert torch.equal(current_actions[1, 0], torch.tensor([2.5]))
    assert torch.equal(next_obs[-1, -1], torch.tensor([12.0, 13.0]))


@pytest.mark.parametrize(
    ("observations", "actions"),
    [
        (torch.zeros(2, 3), torch.zeros(2, 2, 1)),
        (torch.zeros(2, 3, 2), torch.zeros(2, 2)),
        (torch.zeros(2, 3, 2), torch.zeros(3, 2, 1)),
        (torch.zeros(2, 3, 2), torch.zeros(2, 3, 1)),
    ],
)
def test_make_observation_batch_rejects_misaligned_inputs(
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        make_observation_batch(observations, actions)


def test_model_components_return_expected_shapes() -> None:
    model = TinyWorldModel(obs_dim=4, action_dim=2, latent_dim=3, hidden_dim=8)
    observations = torch.randn(5, 6, 4)
    actions = torch.randn(5, 6, 2)

    latents = model.encode(observations)
    reconstruction = model.decode(latents)
    next_latents = model.predict_next_latent(latents, actions)

    assert latents.shape == (5, 6, 3)
    assert reconstruction.shape == observations.shape
    assert next_latents.shape == latents.shape


def test_dynamics_uses_residual_delta_parameterization() -> None:
    model = TinyWorldModel(obs_dim=2, action_dim=1, latent_dim=3, hidden_dim=4)
    for parameter in model.dynamics.parameters():
        parameter.data.zero_()
    latents = torch.randn(2, 3)
    actions = torch.randn(2, 1)

    next_latents = model.predict_next_latent(latents, actions)

    assert torch.equal(next_latents, latents)


def test_forward_returns_world_model_outputs() -> None:
    model = TinyWorldModel(obs_dim=2, action_dim=2, latent_dim=4, hidden_dim=8)
    observations, actions = make_toy_sequences(batch_size=3, horizon=4)
    current_obs, current_actions, _ = make_observation_batch(observations, actions)

    reconstruction, predicted_next_obs, current_z, predicted_next_z = model(
        current_obs,
        current_actions,
    )

    assert reconstruction.shape == current_obs.shape
    assert predicted_next_obs.shape == current_obs.shape
    assert current_z.shape == (3, 4, 4)
    assert predicted_next_z.shape == (3, 4, 4)


def test_rollout_returns_imagined_observations_and_latents() -> None:
    model = TinyWorldModel(obs_dim=2, action_dim=2, latent_dim=4, hidden_dim=8)
    initial_observation = torch.randn(3, 2)
    actions = torch.randn(3, 5, 2)

    predicted_observations, predicted_latents = model.rollout(
        initial_observation,
        actions,
    )

    assert predicted_observations.shape == (3, 6, 2)
    assert predicted_latents.shape == (3, 6, 4)
    assert torch.isfinite(predicted_observations).all()
    assert torch.isfinite(predicted_latents).all()


def test_world_model_loss_returns_scalar_terms() -> None:
    reconstruction = torch.zeros(2, 3, 2)
    current_obs = torch.ones(2, 3, 2)
    predicted_next_obs = torch.full((2, 3, 2), 2.0)
    next_obs = torch.ones(2, 3, 2)
    predicted_next_z = torch.zeros(2, 3, 4)
    target_next_z = torch.ones(2, 3, 4)

    total, reconstruction_loss, prediction_loss, latent_loss = world_model_loss(
        reconstruction,
        current_obs,
        predicted_next_obs,
        next_obs,
        predicted_next_z,
        target_next_z,
        latent_weight=0.5,
    )

    assert total.shape == ()
    assert reconstruction_loss.shape == ()
    assert prediction_loss.shape == ()
    assert latent_loss.shape == ()
    assert torch.allclose(
        total,
        reconstruction_loss + prediction_loss + 0.5 * latent_loss,
    )


def test_gradients_flow_through_world_model_loss() -> None:
    torch.manual_seed(0)
    model = TinyWorldModel(obs_dim=2, action_dim=2, latent_dim=4, hidden_dim=8)
    observations, actions = make_toy_sequences(batch_size=4, horizon=3)
    current_obs, current_actions, next_obs = make_observation_batch(
        observations,
        actions,
    )

    reconstruction, predicted_next_obs, _, predicted_next_z = model(
        current_obs,
        current_actions,
    )
    target_next_z = model.encode(next_obs)
    total, _, _, _ = world_model_loss(
        reconstruction,
        current_obs,
        predicted_next_obs,
        next_obs,
        predicted_next_z,
        target_next_z,
    )
    total.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_model_can_fit_tiny_world_model_loop() -> None:
    torch.manual_seed(1)
    model = TinyWorldModel(obs_dim=2, action_dim=2, latent_dim=4, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    observations, actions = make_toy_sequences(batch_size=64, horizon=5)
    current_obs, current_actions, next_obs = make_observation_batch(
        observations,
        actions,
    )

    with torch.no_grad():
        reconstruction, predicted_next_obs, _, predicted_next_z = model(
            current_obs,
            current_actions,
        )
        target_next_z = model.encode(next_obs)
        initial_loss, _, _, _ = world_model_loss(
            reconstruction,
            current_obs,
            predicted_next_obs,
            next_obs,
            predicted_next_z,
            target_next_z,
        )

    for _ in range(160):
        reconstruction, predicted_next_obs, _, predicted_next_z = model(
            current_obs,
            current_actions,
        )
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

    with torch.no_grad():
        reconstruction, predicted_next_obs, _, predicted_next_z = model(
            current_obs,
            current_actions,
        )
        target_next_z = model.encode(next_obs)
        final_loss, _, _, _ = world_model_loss(
            reconstruction,
            current_obs,
            predicted_next_obs,
            next_obs,
            predicted_next_z,
            target_next_z,
        )

    assert final_loss < initial_loss * 0.2
