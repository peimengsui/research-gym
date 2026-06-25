import pytest
import torch

from implementation import LatentDynamics, dynamics_loss, make_transition_batch


def test_make_transition_batch_slices_aligned_trajectories() -> None:
    latents = torch.tensor(
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

    current_z, current_actions, next_z = make_transition_batch(latents, actions)

    assert current_z.shape == (4, 2)
    assert current_actions.shape == (4, 1)
    assert next_z.shape == (4, 2)
    assert torch.equal(current_z[0], torch.tensor([0.0, 1.0]))
    assert torch.equal(current_actions[2], torch.tensor([2.5]))
    assert torch.equal(next_z[-1], torch.tensor([12.0, 13.0]))


@pytest.mark.parametrize(
    ("latents", "actions"),
    [
        (torch.zeros(2, 3), torch.zeros(2, 2, 1)),
        (torch.zeros(2, 3, 2), torch.zeros(2, 2)),
        (torch.zeros(2, 3, 2), torch.zeros(3, 2, 1)),
        (torch.zeros(2, 3, 2), torch.zeros(2, 3, 1)),
    ],
)
def test_make_transition_batch_rejects_misaligned_inputs(
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        make_transition_batch(latents, actions)


def test_forward_returns_next_latent_shape() -> None:
    model = LatentDynamics(latent_dim=3, action_dim=2, hidden_dim=8)
    z = torch.randn(5, 3)
    actions = torch.randn(5, 2)

    next_z = model(z, actions)

    assert next_z.shape == (5, 3)
    assert torch.isfinite(next_z).all()


def test_forward_uses_residual_delta_parameterization() -> None:
    model = LatentDynamics(latent_dim=2, action_dim=1, hidden_dim=4)
    for parameter in model.network.parameters():
        parameter.data.zero_()
    z = torch.tensor([[1.0, -2.0]])
    action = torch.tensor([[0.5]])

    next_z = model(z, action)

    assert torch.equal(next_z, z)


def test_rollout_returns_initial_state_plus_predicted_future() -> None:
    model = LatentDynamics(latent_dim=2, action_dim=1, hidden_dim=4)
    for parameter in model.network.parameters():
        parameter.data.zero_()
    initial_z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actions = torch.randn(2, 3, 1)

    trajectory = model.rollout(initial_z, actions)

    assert trajectory.shape == (2, 4, 2)
    assert torch.equal(trajectory[:, 0, :], initial_z)
    assert torch.equal(trajectory[:, -1, :], initial_z)


def test_dynamics_loss_returns_scalar_mse() -> None:
    predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 4.0]])

    loss = dynamics_loss(predicted, target)

    assert loss.shape == ()
    assert torch.allclose(loss, torch.tensor(2.0))


def test_gradients_flow_through_one_step_loss() -> None:
    torch.manual_seed(0)
    model = LatentDynamics(latent_dim=3, action_dim=2, hidden_dim=8)
    z = torch.randn(6, 3)
    actions = torch.randn(6, 2)
    target_next_z = z + 0.1

    predicted_next_z = model(z, actions)
    loss = dynamics_loss(predicted_next_z, target_next_z)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_model_can_fit_simple_linear_latent_dynamics() -> None:
    torch.manual_seed(1)
    model = LatentDynamics(latent_dim=2, action_dim=1, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    z = torch.randn(64, 2)
    actions = torch.randn(64, 1)
    target_next_z = z + torch.cat((0.5 * actions, -0.25 * actions), dim=-1)

    with torch.no_grad():
        initial_loss = dynamics_loss(model(z, actions), target_next_z)

    for _ in range(160):
        predicted_next_z = model(z, actions)
        loss = dynamics_loss(predicted_next_z, target_next_z)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = dynamics_loss(model(z, actions), target_next_z)

    assert final_loss < initial_loss * 0.1
