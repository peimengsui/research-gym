import pytest
import torch
from torch import nn

from implementation import (
    OracleNoisePredictor,
    ddim_reverse_step,
    ddim_sample_loop,
    make_ddim_timesteps,
    make_diffusion_schedule,
    predict_x0_from_noise,
    q_sample,
)


class ZeroNoisePredictor(nn.Module):
    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x_t)


def test_make_ddim_timesteps_includes_clean_and_noisiest_endpoints() -> None:
    timesteps = make_ddim_timesteps(num_ddpm_timesteps=10, num_ddim_steps=5)

    assert timesteps.dtype == torch.long
    assert torch.equal(timesteps, torch.tensor([0, 2, 4, 6, 9]))


@pytest.mark.parametrize(
    ("num_ddpm_timesteps", "num_ddim_steps"),
    [(1, 1), (4, 1), (4, 5)],
)
def test_make_ddim_timesteps_rejects_invalid_arguments(
    num_ddpm_timesteps: int,
    num_ddim_steps: int,
) -> None:
    with pytest.raises(ValueError):
        make_ddim_timesteps(num_ddpm_timesteps, num_ddim_steps)


def test_predict_x0_from_noise_recovers_clean_sample() -> None:
    schedule = make_diffusion_schedule(num_timesteps=8, beta_start=0.01, beta_end=0.08)
    x_start = torch.randn(3, 1, 4, 4)
    timesteps = torch.tensor([0, 3, 7])
    noise = torch.randn_like(x_start)
    x_t, returned_noise = q_sample(schedule, x_start, timesteps, noise=noise)

    x0_pred = predict_x0_from_noise(schedule, x_t, timesteps, returned_noise)

    assert torch.allclose(x0_pred, x_start, atol=1e-6)


def test_predict_x0_from_noise_rejects_mismatched_noise_shape() -> None:
    schedule = make_diffusion_schedule(num_timesteps=4)

    with pytest.raises(ValueError):
        predict_x0_from_noise(
            schedule,
            torch.randn(2, 1, 4, 4),
            torch.tensor([0, 1]),
            torch.randn(2, 1, 3, 4),
        )


def test_ddim_reverse_step_eta_zero_is_deterministic() -> None:
    schedule = make_diffusion_schedule(num_timesteps=8, beta_start=0.01, beta_end=0.08)
    x_start = torch.randn(2, 1, 4, 4)
    timesteps = torch.tensor([7, 7])
    previous = torch.tensor([3, 3])
    x_t, _ = q_sample(schedule, x_start, timesteps, noise=torch.randn_like(x_start))
    model = OracleNoisePredictor(schedule, x_start)

    first, first_x0, first_noise = ddim_reverse_step(
        model,
        schedule,
        x_t,
        timesteps,
        previous,
        eta=0.0,
        noise=torch.randn_like(x_t),
    )
    second, second_x0, second_noise = ddim_reverse_step(
        model,
        schedule,
        x_t,
        timesteps,
        previous,
        eta=0.0,
        noise=torch.randn_like(x_t),
    )

    assert torch.allclose(first, second)
    assert torch.allclose(first_x0, second_x0)
    assert torch.allclose(first_noise, second_noise)


def test_ddim_reverse_step_final_step_returns_x0_prediction() -> None:
    schedule = make_diffusion_schedule(num_timesteps=8, beta_start=0.01, beta_end=0.08)
    x_start = torch.randn(2, 1, 4, 4)
    timesteps = torch.tensor([2, 2])
    previous = torch.tensor([-1, -1])
    x_t, _ = q_sample(schedule, x_start, timesteps, noise=torch.randn_like(x_start))
    model = OracleNoisePredictor(schedule, x_start)

    x_previous, x0_pred, _ = ddim_reverse_step(
        model,
        schedule,
        x_t,
        timesteps,
        previous,
        eta=0.0,
    )

    assert torch.allclose(x0_pred, x_start, atol=1e-6)
    assert torch.allclose(x_previous, x0_pred)


def test_ddim_reverse_step_eta_adds_supplied_noise() -> None:
    schedule = make_diffusion_schedule(num_timesteps=8, beta_start=0.01, beta_end=0.08)
    x_t = torch.randn(2, 1, 4, 4)
    timesteps = torch.tensor([7, 7])
    previous = torch.tensor([3, 3])
    model = ZeroNoisePredictor()

    deterministic, _, _ = ddim_reverse_step(
        model,
        schedule,
        x_t,
        timesteps,
        previous,
        eta=0.0,
    )
    stochastic, _, _ = ddim_reverse_step(
        model,
        schedule,
        x_t,
        timesteps,
        previous,
        eta=1.0,
        noise=torch.ones_like(x_t),
    )

    assert stochastic.shape == x_t.shape
    assert not torch.allclose(stochastic, deterministic)


@pytest.mark.parametrize(
    ("timesteps", "previous"),
    [
        (torch.tensor([[3]]), torch.tensor([2])),
        (torch.tensor([3]), torch.tensor([[2]])),
        (torch.tensor([8]), torch.tensor([2])),
        (torch.tensor([3]), torch.tensor([3])),
        (torch.tensor([3]), torch.tensor([-2])),
    ],
)
def test_ddim_reverse_step_rejects_invalid_timestep_inputs(
    timesteps: torch.Tensor,
    previous: torch.Tensor,
) -> None:
    schedule = make_diffusion_schedule(num_timesteps=8)
    model = ZeroNoisePredictor()

    with pytest.raises(ValueError):
        ddim_reverse_step(
            model,
            schedule,
            torch.randn(1, 1, 4, 4),
            timesteps,
            previous,
        )


def test_ddim_sample_loop_returns_shape_and_trajectory() -> None:
    schedule = make_diffusion_schedule(num_timesteps=8)
    model = ZeroNoisePredictor()
    generator = torch.Generator().manual_seed(0)

    sample, trajectory = ddim_sample_loop(
        model,
        schedule,
        shape=(2, 1, 4, 4),
        num_steps=4,
        generator=generator,
        return_all=True,
    )

    assert sample.shape == (2, 1, 4, 4)
    assert len(trajectory) == 5
    assert all(tensor.shape == sample.shape for tensor in trajectory)


def test_ddim_sample_loop_is_deterministic_for_fixed_initial_noise() -> None:
    schedule = make_diffusion_schedule(num_timesteps=8, beta_start=0.01, beta_end=0.08)
    x_start = torch.randn(1, 1, 4, 4)
    timesteps = torch.tensor([7])
    initial_noise = torch.randn_like(x_start)
    x_t, _ = q_sample(schedule, x_start, timesteps, noise=initial_noise)
    model = OracleNoisePredictor(schedule, x_start)

    first = ddim_sample_loop(
        model,
        schedule,
        shape=tuple(x_t.shape),
        num_steps=4,
        eta=0.0,
        initial_noise=x_t,
    )
    second = ddim_sample_loop(
        model,
        schedule,
        shape=tuple(x_t.shape),
        num_steps=4,
        eta=0.0,
        initial_noise=x_t,
    )

    assert torch.allclose(first, second)
    assert torch.allclose(first, x_start, atol=1e-5)


@pytest.mark.parametrize(
    ("shape", "initial_noise"),
    [
        ((1,), None),
        ((0, 1, 4, 4), None),
        ((1, 1, 4, 4), torch.randn(2, 1, 4, 4)),
    ],
)
def test_ddim_sample_loop_rejects_invalid_inputs(
    shape: tuple[int, ...],
    initial_noise: torch.Tensor | None,
) -> None:
    schedule = make_diffusion_schedule(num_timesteps=8)
    model = ZeroNoisePredictor()

    with pytest.raises(ValueError):
        ddim_sample_loop(
            model, schedule, shape=shape, num_steps=4, initial_noise=initial_noise
        )
