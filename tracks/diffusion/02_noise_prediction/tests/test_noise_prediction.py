import pytest
import torch
import torch.nn.functional as F

from implementation import (
    TinyNoisePredictor,
    make_diffusion_schedule,
    make_noise_prediction_batch,
    make_toy_points,
    noise_prediction_loss,
    q_sample,
    sinusoidal_timestep_embedding,
)


def test_sinusoidal_timestep_embedding_shape_and_values() -> None:
    timesteps = torch.tensor([0, 1, 2])

    embedding = sinusoidal_timestep_embedding(timesteps, dim=4)

    assert embedding.shape == (3, 4)
    assert torch.allclose(embedding[0, :2], torch.zeros(2))
    assert torch.allclose(embedding[0, 2:], torch.ones(2))
    assert not torch.allclose(embedding[1], embedding[2])


def test_sinusoidal_timestep_embedding_pads_odd_dimension() -> None:
    embedding = sinusoidal_timestep_embedding(torch.tensor([3, 4]), dim=5)

    assert embedding.shape == (2, 5)
    assert torch.equal(embedding[:, -1], torch.zeros(2))


@pytest.mark.parametrize(
    ("timesteps", "dim"),
    [(torch.tensor([[1]]), 4), (torch.tensor([1]), 0)],
)
def test_sinusoidal_timestep_embedding_rejects_invalid_inputs(
    timesteps: torch.Tensor,
    dim: int,
) -> None:
    with pytest.raises(ValueError):
        sinusoidal_timestep_embedding(timesteps, dim)


def test_tiny_noise_predictor_returns_noise_shape() -> None:
    model = TinyNoisePredictor(data_dim=2, time_embed_dim=8, hidden_dim=16)
    x_t = torch.randn(5, 2)
    timesteps = torch.tensor([0, 1, 2, 3, 4])

    predicted_noise = model(x_t, timesteps)

    assert predicted_noise.shape == x_t.shape


@pytest.mark.parametrize(
    ("data_dim", "time_embed_dim", "hidden_dim"),
    [(0, 8, 16), (2, 0, 16), (2, 8, 0)],
)
def test_tiny_noise_predictor_rejects_invalid_constructor_args(
    data_dim: int,
    time_embed_dim: int,
    hidden_dim: int,
) -> None:
    with pytest.raises(ValueError):
        TinyNoisePredictor(data_dim, time_embed_dim, hidden_dim)


def test_q_sample_returns_known_noise_target() -> None:
    schedule = make_diffusion_schedule(num_timesteps=4, beta_start=0.1, beta_end=0.2)
    x_start = torch.randn(3, 2)
    timesteps = torch.tensor([0, 1, 3])
    noise = torch.randn_like(x_start)

    x_t, returned_noise = q_sample(schedule, x_start, timesteps, noise=noise)

    assert x_t.shape == x_start.shape
    assert torch.equal(returned_noise, noise)


def test_make_noise_prediction_batch_has_expected_shapes_and_ranges() -> None:
    schedule = make_diffusion_schedule(num_timesteps=10)
    data = make_toy_points(16)
    generator = torch.Generator().manual_seed(0)

    batch = make_noise_prediction_batch(
        schedule,
        data,
        batch_size=8,
        generator=generator,
    )

    assert batch.x_t.shape == (8, 2)
    assert batch.timesteps.shape == (8,)
    assert batch.noise.shape == (8, 2)
    assert batch.x_start.shape == (8, 2)
    assert int(batch.timesteps.min()) >= 0
    assert int(batch.timesteps.max()) < schedule.num_timesteps


@pytest.mark.parametrize(
    ("data", "batch_size"),
    [(torch.randn(4, 2, 1), 2), (torch.randn(4, 2), 0)],
)
def test_make_noise_prediction_batch_rejects_invalid_inputs(
    data: torch.Tensor,
    batch_size: int,
) -> None:
    schedule = make_diffusion_schedule(num_timesteps=4)

    with pytest.raises(ValueError):
        make_noise_prediction_batch(schedule, data, batch_size)


def test_noise_prediction_loss_matches_manual_mse() -> None:
    class ConstantPredictor(TinyNoisePredictor):
        def __init__(self):
            super().__init__(data_dim=2, time_embed_dim=4, hidden_dim=4)

        def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x_t) * 0.25

    model = ConstantPredictor()
    x_t = torch.zeros(3, 2)
    timesteps = torch.tensor([0, 1, 2])
    noise = torch.tensor([[0.0, 0.5], [0.25, 0.25], [1.0, -1.0]])

    actual = noise_prediction_loss(model, x_t, timesteps, noise)
    expected = F.mse_loss(torch.ones_like(noise) * 0.25, noise)

    assert torch.allclose(actual, expected)


def test_noise_prediction_loss_rejects_mismatched_noise_shape() -> None:
    model = TinyNoisePredictor(data_dim=2, time_embed_dim=4, hidden_dim=8)

    with pytest.raises(ValueError):
        noise_prediction_loss(
            model,
            torch.randn(3, 2),
            torch.tensor([0, 1, 2]),
            torch.randn(3, 3),
        )


def test_gradients_flow_through_noise_predictor() -> None:
    torch.manual_seed(1)
    model = TinyNoisePredictor(data_dim=2, time_embed_dim=8, hidden_dim=16)
    schedule = make_diffusion_schedule(num_timesteps=8)
    batch = make_noise_prediction_batch(schedule, make_toy_points(16), batch_size=8)

    loss = noise_prediction_loss(model, batch.x_t, batch.timesteps, batch.noise)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_tiny_noise_predictor_can_fit_fixed_batch() -> None:
    torch.manual_seed(2)
    schedule = make_diffusion_schedule(num_timesteps=12)
    data = make_toy_points(32)
    generator = torch.Generator().manual_seed(3)
    batch = make_noise_prediction_batch(
        schedule,
        data,
        batch_size=64,
        generator=generator,
    )
    model = TinyNoisePredictor(data_dim=2, time_embed_dim=16, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)

    with torch.no_grad():
        initial_loss = noise_prediction_loss(
            model,
            batch.x_t,
            batch.timesteps,
            batch.noise,
        )

    for _ in range(160):
        loss = noise_prediction_loss(model, batch.x_t, batch.timesteps, batch.noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = noise_prediction_loss(
            model,
            batch.x_t,
            batch.timesteps,
            batch.noise,
        )

    assert final_loss < initial_loss * 0.35
