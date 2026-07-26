"""Reference solution for the flow matching lesson."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FlowMatchingBatch:
    """One straight-path flow-matching training batch.

    Shapes:
        path_points: (batch, channels, height, width)
        times: (batch,)
        target_velocity: (batch, channels, height, width)
        data: (batch, channels, height, width)
        noise: (batch, channels, height, width)
    """

    path_points: torch.Tensor
    times: torch.Tensor
    target_velocity: torch.Tensor
    data: torch.Tensor
    noise: torch.Tensor


def _broadcast_times(times: torch.Tensor, data_shape: torch.Size) -> torch.Tensor:
    """Reshape one time per example for broadcasting over data dimensions."""

    if times.ndim != 1 or times.shape[0] != data_shape[0]:
        raise ValueError("times must have shape (batch,)")
    return times.reshape((times.shape[0],) + (1,) * (len(data_shape) - 1))


def linear_interpolation(
    noise: torch.Tensor,
    data: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Return x_t = (1 - t) * noise + t * data."""

    if noise.shape != data.shape:
        raise ValueError("noise and data must have the same shape")
    if data.ndim < 2:
        raise ValueError("data must have shape (batch, ...)")
    if torch.any(times < 0.0) or torch.any(times > 1.0):
        raise ValueError("times must be between 0 and 1")
    t = _broadcast_times(times.to(device=data.device, dtype=data.dtype), data.shape)
    return (1.0 - t) * noise + t * data


def straight_path_velocity(
    noise: torch.Tensor,
    data: torch.Tensor,
) -> torch.Tensor:
    """Return the constant derivative dx_t/dt = data - noise."""

    if noise.shape != data.shape:
        raise ValueError("noise and data must have the same shape")
    return data - noise


def make_flow_matching_batch(
    data: torch.Tensor,
    times: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowMatchingBatch:
    """Pair data with noise and sample points along the straight path."""

    if data.ndim != 4:
        raise ValueError("data must have shape (batch, channels, height, width)")
    if times is None:
        times = torch.rand(
            data.shape[0],
            device=data.device,
            dtype=data.dtype,
            generator=generator,
        )
    if noise is None:
        noise = torch.randn(
            data.shape,
            device=data.device,
            dtype=data.dtype,
            generator=generator,
        )
    path_points = linear_interpolation(noise, data, times)
    velocity = straight_path_velocity(noise, data)
    return FlowMatchingBatch(
        path_points=path_points,
        times=times,
        target_velocity=velocity,
        data=data,
        noise=noise,
    )


def sinusoidal_time_embedding(
    times: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Embed continuous times in [0, 1] into vectors of shape (batch, dim)."""

    if times.ndim != 1:
        raise ValueError("times must have shape (batch,)")
    if dim <= 0:
        raise ValueError("dim must be positive")
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros((times.shape[0], 1), device=times.device)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=times.device, dtype=torch.float32)
        / half_dim
    )
    angles = 2.0 * math.pi * times.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    return F.pad(embedding, (0, dim % 2))


class TimeConditionedResidualBlock(nn.Module):
    """A residual convolution block with additive continuous-time conditioning."""

    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embed_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("x has the wrong image shape or channel count")
        if time_embedding.shape != (x.shape[0], self.time_embed_dim):
            raise ValueError("time_embedding must have shape (batch, time_embed_dim)")
        time_bias = self.time_projection(time_embedding)[:, :, None, None]
        return F.silu(self.conv2(F.silu(self.conv1(x) + time_bias)) + x)


class TinyVelocityModel(nn.Module):
    """A tiny continuous-time velocity field for image-shaped tensors."""

    def __init__(
        self,
        channels: int = 1,
        hidden_channels: int = 16,
        time_embed_dim: int = 16,
    ):
        super().__init__()
        if channels <= 0 or hidden_channels <= 0 or time_embed_dim <= 0:
            raise ValueError("channel counts and time_embed_dim must be positive")
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.input_conv = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.block1 = TimeConditionedResidualBlock(hidden_channels, time_embed_dim)
        self.block2 = TimeConditionedResidualBlock(hidden_channels, time_embed_dim)
        self.output_conv = nn.Conv2d(
            hidden_channels, channels, kernel_size=3, padding=1
        )

    def forward(self, x_t: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4 or x_t.shape[1] != self.channels:
            raise ValueError("x_t has the wrong image shape or channel count")
        if times.shape != (x_t.shape[0],):
            raise ValueError("times must have shape (batch,)")
        time_embedding = sinusoidal_time_embedding(times, self.time_embed_dim).to(
            device=x_t.device, dtype=x_t.dtype
        )
        time_embedding = self.time_mlp(time_embedding)
        h = F.silu(self.input_conv(x_t))
        h = self.block1(h, time_embedding)
        h = self.block2(h, time_embedding)
        return self.output_conv(h)


def flow_matching_loss(
    model: nn.Module,
    data: torch.Tensor,
    times: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Train a model to predict the straight-path velocity at sampled points."""

    batch = make_flow_matching_batch(data, times, noise, generator)
    predicted_velocity = model(batch.path_points, batch.times)
    return F.mse_loss(predicted_velocity, batch.target_velocity)


def _time_batch(
    x: torch.Tensor,
    time: float | torch.Tensor,
) -> torch.Tensor:
    """Convert a scalar or batch of times to shape (batch,) on x's device."""

    times = torch.as_tensor(time, device=x.device, dtype=x.dtype)
    if times.ndim == 0:
        times = times.expand(x.shape[0])
    if times.shape != (x.shape[0],):
        raise ValueError("time must be a scalar or have shape (batch,)")
    return times


@torch.no_grad()
def euler_step(
    model: nn.Module,
    x: torch.Tensor,
    time: float | torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Advance x by one explicit Euler ODE step."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    times = _time_batch(x, time)
    velocity = model(x, times)
    if velocity.shape != x.shape:
        raise ValueError("model velocity must have the same shape as x")
    return x + step_size * velocity


@torch.no_grad()
def midpoint_step(
    model: nn.Module,
    x: torch.Tensor,
    time: float | torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Advance x with the second-order explicit midpoint method."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    times = _time_batch(x, time)
    first_velocity = model(x, times)
    midpoint = x + 0.5 * step_size * first_velocity
    midpoint_times = times + 0.5 * step_size
    midpoint_velocity = model(midpoint, midpoint_times)
    if midpoint_velocity.shape != x.shape:
        raise ValueError("model velocity must have the same shape as x")
    return x + step_size * midpoint_velocity


@torch.no_grad()
def ode_sample(
    model: nn.Module,
    shape: tuple[int, ...],
    num_steps: int,
    solver: str = "euler",
    initial_noise: torch.Tensor | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    return_all: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Integrate dx/dt = v_theta(x, t) from noise at t=0 to data at t=1."""

    if len(shape) < 2 or shape[0] <= 0:
        raise ValueError("shape must include a positive batch dimension")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if solver not in {"euler", "midpoint"}:
        raise ValueError("solver must be 'euler' or 'midpoint'")
    device = torch.device("cpu") if device is None else device
    if initial_noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        if initial_noise.shape != shape:
            raise ValueError("initial_noise must match shape")
        x = initial_noise.to(device)
    trajectory = [x.clone()] if return_all else []
    step_size = 1.0 / num_steps
    step_function = euler_step if solver == "euler" else midpoint_step
    for step in range(num_steps):
        time = step * step_size
        x = step_function(model, x, time, step_size)
        if return_all:
            trajectory.append(x.clone())
    if return_all:
        return x, trajectory
    return x


class TargetVelocityField(nn.Module):
    """Debug oracle whose ODE flows from any current point to a known target."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.register_buffer("target", target)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        target = self.target.to(device=x.device, dtype=x.dtype)
        if target.shape[0] == 1 and x.shape[0] > 1:
            target = target.expand_as(x)
        if target.shape != x.shape:
            raise ValueError("target must match x, or have batch size 1")
        remaining_time = _broadcast_times(1.0 - times, x.shape)
        return (target - x) / remaining_time.clamp_min(1e-6)
