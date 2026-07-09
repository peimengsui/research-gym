import pytest
import torch
from torch import nn

from implementation import (
    OracleNoisePredictor,
    ddpm_reverse_mean,
    ddpm_reverse_step,
    ddpm_sample_loop,
    gather_by_timestep,
    make_diffusion_schedule,
    make_toy_points,
    predict_x0_from_noise,
    q_sample,
)


class ZeroNoisePredictor(nn.Module):
    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x_t)


def test_schedule_includes_posterior_variance() -> None:
    schedule = make_diffusion_schedule(
        num_timesteps=4,
        beta_start=0.1,
        beta_end=0.4,
    )

    expected_prev = torch.tensor([1.0, 0.9, 0.72, 0.504])
    expected_variance = (
        schedule.betas * (1.0 - schedule.alpha_bars_prev) / (1.0 - schedule.alpha_bars)
    )

    assert torch.allclose(schedule.alpha_bars_prev, expected_prev)
    assert torch.allclose(schedule.posterior_variance, expected_variance)
    assert schedule.posterior_variance[0] == 0.0


def test_predict_x0_from_noise_recovers_clean_data_with_true_noise() -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)
    x_start = make_toy_points(4)
    timesteps = torch.tensor([0, 1, 3, 4])
    noise = torch.randn_like(x_start)
    x_t, true_noise = q_sample(schedule, x_start, timesteps, noise=noise)

    predicted_x0 = predict_x0_from_noise(schedule, x_t, timesteps, true_noise)

    assert torch.allclose(predicted_x0, x_start, atol=1e-5)


def test_predict_x0_rejects_mismatched_noise_shape() -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)

    with pytest.raises(ValueError):
        predict_x0_from_noise(
            schedule,
            torch.randn(2, 3),
            torch.tensor([0, 1]),
            torch.randn(2, 4),
        )


def test_ddpm_reverse_mean_reconstructs_x0_at_final_step_with_true_noise() -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)
    x_start = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    timesteps = torch.tensor([0, 0])
    noise = torch.randn_like(x_start)
    x_t, true_noise = q_sample(schedule, x_start, timesteps, noise=noise)

    mean = ddpm_reverse_mean(schedule, x_t, timesteps, true_noise)

    assert torch.allclose(mean, x_start, atol=1e-5)


def test_ddpm_reverse_step_skips_noise_at_timestep_zero() -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)
    x_t = torch.randn(3, 2)
    timesteps = torch.zeros(3, dtype=torch.long)
    injected_noise = torch.randn_like(x_t) * 10.0

    x_previous, mean = ddpm_reverse_step(
        ZeroNoisePredictor(),
        schedule,
        x_t,
        timesteps,
        noise=injected_noise,
    )

    assert torch.allclose(x_previous, mean)


def test_ddpm_reverse_step_adds_posterior_noise_when_timestep_is_nonzero() -> None:
    schedule = make_diffusion_schedule(
        num_timesteps=5,
        beta_start=0.1,
        beta_end=0.2,
    )
    x_t = torch.zeros(2, 2)
    timesteps = torch.tensor([2, 4])
    noise = torch.ones_like(x_t)

    x_previous, mean = ddpm_reverse_step(
        ZeroNoisePredictor(),
        schedule,
        x_t,
        timesteps,
        noise=noise,
    )
    variance = gather_by_timestep(schedule.posterior_variance, timesteps, x_t.shape)

    assert torch.allclose(x_previous, mean + torch.sqrt(variance))


@pytest.mark.parametrize(
    ("x_t", "timesteps", "noise"),
    [
        (torch.randn(2), torch.tensor([0, 1]), None),
        (torch.randn(2, 3), torch.tensor([[0], [1]]), None),
        (torch.randn(2, 3), torch.tensor([0.0, 1.0]), None),
        (torch.randn(2, 3), torch.tensor([0, 99]), None),
        (torch.randn(2, 3), torch.tensor([0, 1]), torch.randn(2, 4)),
    ],
)
def test_ddpm_reverse_step_rejects_invalid_shapes_or_timesteps(
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None,
) -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)

    with pytest.raises(ValueError):
        ddpm_reverse_step(
            ZeroNoisePredictor(),
            schedule,
            x_t,
            timesteps,
            noise=noise,
        )


def test_oracle_noise_predictor_matches_known_forward_noise() -> None:
    schedule = make_diffusion_schedule(num_timesteps=6)
    x_start = make_toy_points(3)
    timesteps = torch.tensor([1, 3, 5])
    noise = torch.randn_like(x_start)
    x_t, true_noise = q_sample(schedule, x_start, timesteps, noise=noise)
    oracle = OracleNoisePredictor(schedule, x_start)

    predicted_noise = oracle(x_t, timesteps)

    assert torch.allclose(predicted_noise, true_noise, atol=1e-5)


def test_ddpm_sample_loop_returns_expected_shape_and_trajectory() -> None:
    schedule = make_diffusion_schedule(num_timesteps=4)
    generator = torch.Generator().manual_seed(0)

    final, trajectory = ddpm_sample_loop(
        ZeroNoisePredictor(),
        schedule,
        shape=(3, 2),
        generator=generator,
        return_all=True,
    )

    assert final.shape == (3, 2)
    assert len(trajectory) == schedule.num_timesteps + 1
    assert all(sample.shape == (3, 2) for sample in trajectory)
    assert torch.equal(final, trajectory[-1])


def test_ddpm_sample_loop_rejects_invalid_shape() -> None:
    schedule = make_diffusion_schedule(num_timesteps=4)

    with pytest.raises(ValueError):
        ddpm_sample_loop(ZeroNoisePredictor(), schedule, shape=(2,))
