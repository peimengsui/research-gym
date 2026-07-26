import math

import torch
from torch import nn

from implementation import (
    TargetVelocityField,
    TinyVelocityModel,
    euler_step,
    flow_matching_loss,
    linear_interpolation,
    make_flow_matching_batch,
    ode_sample,
    straight_path_velocity,
)


def test_linear_path_has_noise_data_endpoints_and_correct_midpoint() -> None:
    noise = torch.zeros(3, 1, 2, 2)
    data = torch.full_like(noise, 4.0)
    times = torch.tensor([0.0, 0.5, 1.0])

    points = linear_interpolation(noise, data, times)

    assert torch.equal(points[0], noise[0])
    assert torch.equal(points[1], torch.full_like(points[1], 2.0))
    assert torch.equal(points[2], data[2])


def test_straight_path_velocity_matches_finite_difference() -> None:
    torch.manual_seed(9)
    noise = torch.randn(2, 1, 4, 4)
    data = torch.randn(2, 1, 4, 4)
    times = torch.tensor([0.2, 0.7])
    delta = 1e-3

    point = linear_interpolation(noise, data, times)
    later = linear_interpolation(noise, data, times + delta)
    finite_difference = (later - point) / delta

    assert torch.allclose(
        straight_path_velocity(noise, data), finite_difference, atol=2e-4
    )


def test_flow_batch_has_expected_shapes() -> None:
    data = torch.randn(4, 2, 6, 6)

    batch = make_flow_matching_batch(data)

    assert batch.path_points.shape == data.shape
    assert batch.target_velocity.shape == data.shape
    assert batch.noise.shape == data.shape
    assert batch.times.shape == (4,)
    assert torch.all((batch.times >= 0.0) & (batch.times <= 1.0))


def test_velocity_model_and_loss_backpropagate() -> None:
    torch.manual_seed(9)
    model = TinyVelocityModel(channels=1, hidden_channels=8, time_embed_dim=8)
    data = torch.randn(3, 1, 8, 8)

    loss = flow_matching_loss(model, data)
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


class ConstantField(nn.Module):
    def __init__(self, velocity: float):
        super().__init__()
        self.velocity = velocity

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.velocity)


def test_euler_step_integrates_constant_velocity() -> None:
    x = torch.zeros(2, 1, 2, 2)

    updated = euler_step(ConstantField(3.0), x, time=0.25, step_size=0.2)

    assert torch.allclose(updated, torch.full_like(x, 0.6))


class ExponentialField(nn.Module):
    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return x


def test_midpoint_is_more_accurate_than_euler_for_curved_trajectory() -> None:
    initial = torch.ones(1, 1)
    euler = ode_sample(
        ExponentialField(),
        shape=(1, 1),
        num_steps=4,
        solver="euler",
        initial_noise=initial,
    )
    midpoint = ode_sample(
        ExponentialField(),
        shape=(1, 1),
        num_steps=4,
        solver="midpoint",
        initial_noise=initial,
    )
    exact = torch.tensor([[math.e]])

    assert torch.abs(midpoint - exact).item() < torch.abs(euler - exact).item()


def test_oracle_ode_sampling_reaches_target_and_returns_trajectory() -> None:
    torch.manual_seed(9)
    target = torch.full((1, 1, 8, 8), -1.0)
    target[:, :, 2:6, 2:6] = 1.0
    initial_noise = torch.randn_like(target)
    oracle = TargetVelocityField(target)

    sample, trajectory = ode_sample(
        oracle,
        shape=tuple(target.shape),
        num_steps=5,
        solver="euler",
        initial_noise=initial_noise,
        return_all=True,
    )

    assert torch.allclose(sample, target, atol=1e-5)
    assert len(trajectory) == 6
    assert torch.equal(trajectory[0], initial_noise)
