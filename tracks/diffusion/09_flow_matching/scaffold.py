"""Learner scaffold for straight-path flow matching and ODE sampling.

The tiny velocity network is provided. Your TODOs focus on the probability
path, velocity target, training objective, and numerical integration.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FlowMatchingBatch:
    """One flow-matching batch with image tensors shaped (B, C, H, W)."""

    path_points: torch.Tensor
    times: torch.Tensor
    target_velocity: torch.Tensor
    data: torch.Tensor
    noise: torch.Tensor


def _broadcast_times(times: torch.Tensor, data_shape: torch.Size) -> torch.Tensor:
    if times.ndim != 1 or times.shape[0] != data_shape[0]:
        raise ValueError("times must have shape (batch,)")
    return times.reshape((times.shape[0],) + (1,) * (len(data_shape) - 1))


def linear_interpolation(
    noise: torch.Tensor,
    data: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Return path points shaped like data, with noise at t=0 and data at t=1."""

    if noise.shape != data.shape:
        raise ValueError("noise and data must have the same shape")
    if torch.any(times < 0.0) or torch.any(times > 1.0):
        raise ValueError("times must be between 0 and 1")
    # TODO 1: Reshape times for broadcasting and compute
    # x_t = (1 - t) * noise + t * data.
    raise NotImplementedError


def straight_path_velocity(
    noise: torch.Tensor,
    data: torch.Tensor,
) -> torch.Tensor:
    """Return dx_t/dt with the same shape as noise and data."""

    if noise.shape != data.shape:
        raise ValueError("noise and data must have the same shape")
    # TODO 2: Differentiate the straight interpolation with respect to t.
    raise NotImplementedError


def make_flow_matching_batch(
    data: torch.Tensor,
    times: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowMatchingBatch:
    """Provided batch assembly around your path and velocity functions."""

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
    return FlowMatchingBatch(
        path_points=linear_interpolation(noise, data, times),
        times=times,
        target_velocity=straight_path_velocity(noise, data),
        data=data,
        noise=noise,
    )


def sinusoidal_time_embedding(
    times: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Provided continuous-time embedding with shape (batch, dim)."""

    if times.ndim != 1 or dim <= 0:
        raise ValueError("times must be (batch,) and dim must be positive")
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros((times.shape[0], 1), device=times.device)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=times.device, dtype=torch.float32)
        / half_dim
    )
    angles = 2.0 * math.pi * times.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return F.pad(
        torch.cat((torch.sin(angles), torch.cos(angles)), dim=1),
        (0, dim % 2),
    )


class TimeConditionedResidualBlock(nn.Module):
    """Provided residual block with continuous-time conditioning."""

    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embed_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        time_bias = self.time_projection(time_embedding)[:, :, None, None]
        return F.silu(self.conv2(F.silu(self.conv1(x) + time_bias)) + x)


class TinyVelocityModel(nn.Module):
    """Provided network for v_theta(x_t, t), returning the same shape as x_t."""

    def __init__(
        self,
        channels: int = 1,
        hidden_channels: int = 16,
        time_embed_dim: int = 16,
    ):
        super().__init__()
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
    """Return scalar MSE between predicted and target path velocity."""

    # TODO 3: Build a flow-matching batch, predict velocity at its path points
    # and times, and compare with target_velocity using mean-squared error.
    raise NotImplementedError


def _time_batch(
    x: torch.Tensor,
    time: float | torch.Tensor,
) -> torch.Tensor:
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
    """Advance one step with x_next = x + step_size * v(x, t)."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    # TODO 4: Build a batch of times, evaluate the velocity once, and take the
    # explicit Euler update.
    raise NotImplementedError


@torch.no_grad()
def midpoint_step(
    model: nn.Module,
    x: torch.Tensor,
    time: float | torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Advance one step with the second-order explicit midpoint solver."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    # TODO 5: Use one velocity evaluation to estimate the state at the middle
    # of the interval, evaluate velocity there at t + step_size/2, and use that
    # second velocity for the full update.
    raise NotImplementedError


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
    """Integrate from Gaussian noise at t=0 to a generated sample at t=1."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if solver not in {"euler", "midpoint"}:
        raise ValueError("solver must be 'euler' or 'midpoint'")
    # TODO 6:
    # 1. Start from initial_noise or sample Gaussian noise.
    # 2. Set step_size = 1 / num_steps.
    # 3. Apply the selected solver at t = step * step_size.
    # 4. Optionally collect the initial state and every updated state.
    raise NotImplementedError


class TargetVelocityField(nn.Module):
    """Provided oracle that flows from any current state to a known target."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.register_buffer("target", target)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        target = self.target.to(device=x.device, dtype=x.dtype)
        if target.shape[0] == 1 and x.shape[0] > 1:
            target = target.expand_as(x)
        remaining_time = _broadcast_times(1.0 - times, x.shape)
        return (target - x) / remaining_time.clamp_min(1e-6)
