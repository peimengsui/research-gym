import pytest
import torch

from implementation import (
    gather_by_timestep,
    linear_beta_schedule,
    make_diffusion_schedule,
    make_toy_points,
    q_sample,
    sample_timesteps,
)


def test_linear_beta_schedule_has_expected_values() -> None:
    betas = linear_beta_schedule(num_timesteps=5, beta_start=0.1, beta_end=0.5)

    assert betas.shape == (5,)
    assert torch.allclose(betas, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert torch.all((0.0 < betas) & (betas < 1.0))


@pytest.mark.parametrize(
    ("num_timesteps", "beta_start", "beta_end"),
    [(0, 0.1, 0.2), (5, 0.0, 0.2), (5, 0.1, 1.0), (5, 0.3, 0.2)],
)
def test_linear_beta_schedule_rejects_invalid_arguments(
    num_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> None:
    with pytest.raises(ValueError):
        linear_beta_schedule(num_timesteps, beta_start, beta_end)


def test_make_diffusion_schedule_precomputes_monotonic_values() -> None:
    schedule = make_diffusion_schedule(
        num_timesteps=4,
        beta_start=0.1,
        beta_end=0.4,
    )

    assert torch.allclose(schedule.alphas, 1.0 - schedule.betas)
    assert torch.allclose(
        schedule.alpha_bars,
        torch.tensor([0.9, 0.72, 0.504, 0.3024]),
    )
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
    assert torch.allclose(schedule.sqrt_alpha_bars**2, schedule.alpha_bars)
    assert torch.allclose(
        schedule.sqrt_one_minus_alpha_bars**2,
        1.0 - schedule.alpha_bars,
    )
    assert schedule.num_timesteps == 4


def test_gather_by_timestep_broadcasts_to_sample_shape() -> None:
    values = torch.tensor([10.0, 20.0, 30.0, 40.0])
    timesteps = torch.tensor([2, 0, 3])

    gathered = gather_by_timestep(values, timesteps, broadcast_shape=(3, 2, 4, 4))

    assert gathered.shape == (3, 1, 1, 1)
    assert torch.equal(gathered[:, 0, 0, 0], torch.tensor([30.0, 10.0, 40.0]))


def test_q_sample_uses_closed_form_coefficients_with_supplied_noise() -> None:
    schedule = make_diffusion_schedule(
        num_timesteps=3,
        beta_start=0.1,
        beta_end=0.3,
    )
    x_start = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    noise = torch.tensor([[0.5, 0.25], [-0.5, -0.25]])
    timesteps = torch.tensor([0, 2])

    x_t, returned_noise = q_sample(schedule, x_start, timesteps, noise=noise)

    expected = (
        schedule.sqrt_alpha_bars[timesteps].reshape(2, 1) * x_start
        + schedule.sqrt_one_minus_alpha_bars[timesteps].reshape(2, 1) * noise
    )
    assert torch.allclose(x_t, expected)
    assert torch.equal(returned_noise, noise)


def test_q_sample_supports_image_shaped_tensors() -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)
    x_start = torch.ones(2, 1, 3, 3)
    timesteps = torch.tensor([1, 4])

    x_t, noise = q_sample(schedule, x_start, timesteps)

    assert x_t.shape == x_start.shape
    assert noise.shape == x_start.shape
    assert not torch.equal(x_t[0], x_t[1])


def test_q_sample_preserves_gradient_flow_to_clean_input() -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)
    x_start = torch.randn(3, 2, requires_grad=True)
    noise = torch.zeros_like(x_start)
    timesteps = torch.tensor([0, 1, 2])

    x_t, _ = q_sample(schedule, x_start, timesteps, noise=noise)
    x_t.sum().backward()

    assert x_start.grad is not None
    assert torch.all(x_start.grad > 0)


@pytest.mark.parametrize(
    ("x_start", "timesteps", "noise"),
    [
        (torch.randn(2), torch.tensor([0, 1]), None),
        (torch.randn(2, 3), torch.tensor([[0], [1]]), None),
        (torch.randn(2, 3), torch.tensor([0.0, 1.0]), None),
        (torch.randn(2, 3), torch.tensor([0, 99]), None),
        (torch.randn(2, 3), torch.tensor([0, 1]), torch.randn(2, 4)),
    ],
)
def test_q_sample_rejects_invalid_shapes_or_timesteps(
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None,
) -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)

    with pytest.raises(ValueError):
        q_sample(schedule, x_start, timesteps, noise=noise)


def test_sample_timesteps_returns_valid_integer_batch() -> None:
    timesteps = sample_timesteps(batch_size=32, num_timesteps=10)

    assert timesteps.shape == (32,)
    assert timesteps.dtype == torch.int64
    assert int(timesteps.min()) >= 0
    assert int(timesteps.max()) < 10


def test_make_toy_points_returns_unit_circle_points() -> None:
    points = make_toy_points(8)

    assert points.shape == (8, 2)
    assert torch.allclose(points.norm(dim=1), torch.ones(8), atol=1e-6)
