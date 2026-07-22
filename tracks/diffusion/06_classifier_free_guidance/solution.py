"""Reference solution for the classifier-free guidance lesson."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar schedules for noising and DDIM sampling."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


def make_diffusion_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """Build a linear diffusion schedule."""

    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if not (0.0 < beta_start <= beta_end < 1.0):
        raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


def gather_by_timestep(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Gather one schedule scalar per batch item and make it broadcastable."""

    if values.ndim != 1:
        raise ValueError("values must have shape (num_timesteps,)")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,)")
    if len(broadcast_shape) < 1 or broadcast_shape[0] != timesteps.shape[0]:
        raise ValueError("broadcast_shape must have the same batch size")
    gathered = values[timesteps.to(device=values.device, dtype=torch.long)]
    return gathered.reshape((timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1))


def q_sample(
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon."""

    if x_start.ndim < 2:
        raise ValueError("x_start must have shape (batch, ...)")
    if timesteps.shape != (x_start.shape[0],):
        raise ValueError("timesteps must have shape (batch,)")
    if torch.any(timesteps < 0) or torch.any(timesteps >= schedule.num_timesteps):
        raise ValueError("timesteps out of range")
    if noise is None:
        noise = torch.randn_like(x_start)
    if noise.shape != x_start.shape:
        raise ValueError("noise must have the same shape as x_start")
    sqrt_alpha_bar = gather_by_timestep(
        schedule.sqrt_alpha_bars.to(x_start.device), timesteps, x_start.shape
    )
    sqrt_one_minus_alpha_bar = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars.to(x_start.device),
        timesteps,
        x_start.shape,
    )
    return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise, noise


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Return sinusoidal timestep embeddings with shape (batch, dim)."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,)")
    if dim <= 0:
        raise ValueError("dim must be positive")
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros((timesteps.shape[0], 1), device=timesteps.device)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / half_dim
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)
    return F.pad(embedding, (0, dim % 2))


class ConditionedConvBlock(nn.Module):
    """Two convolution layers with one additive conditioning vector."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.condition_projection = nn.Linear(condition_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("x has the wrong image shape or channel count")
        if condition.shape != (x.shape[0], self.condition_dim):
            raise ValueError("condition must have shape (batch, condition_dim)")
        h = self.conv1(x)
        bias = self.condition_projection(condition)[:, :, None, None]
        return F.silu(self.conv2(F.silu(h + bias)))


class ConditionalTinyUNet(nn.Module):
    """A tiny U-Net conditioned on timestep and discrete class label."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 8,
        condition_dim: int = 16,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if in_channels <= 0 or base_channels <= 0 or condition_dim <= 0:
            raise ValueError("channel counts and condition_dim must be positive")
        self.num_classes = num_classes
        self.null_class = num_classes
        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim),
        )
        self.class_embedding = nn.Embedding(num_classes + 1, condition_dim)
        self.input_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        self.down_block = ConditionedConvBlock(
            base_channels, base_channels * 2, condition_dim
        )
        self.bottleneck = ConditionedConvBlock(
            base_channels * 2, base_channels * 2, condition_dim
        )
        self.up_block = ConditionedConvBlock(
            base_channels * 4, base_channels, condition_dim
        )
        self.output_conv = nn.Conv2d(
            base_channels, in_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 4 or x_t.shape[1] != self.in_channels:
            raise ValueError("x_t has the wrong image shape or channel count")
        if x_t.shape[-2] % 2 != 0 or x_t.shape[-1] % 2 != 0:
            raise ValueError("height and width must be even")
        if timesteps.shape != (x_t.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")
        if class_labels.shape != (x_t.shape[0],):
            raise ValueError("class_labels must have shape (batch,)")
        if class_labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("class_labels must be an integer tensor")
        if torch.any(class_labels < 0) or torch.any(class_labels > self.null_class):
            raise ValueError("class_labels out of range")

        time_condition = sinusoidal_timestep_embedding(
            timesteps, self.condition_dim
        ).to(device=x_t.device, dtype=x_t.dtype)
        time_condition = self.time_mlp(time_condition)
        class_condition = self.class_embedding(class_labels.to(x_t.device)).to(
            dtype=x_t.dtype
        )
        condition = time_condition + class_condition

        h = F.silu(self.input_conv(x_t))
        skip = self.down_block(h, condition)
        h = F.avg_pool2d(skip, kernel_size=2)
        h = self.bottleneck(h, condition)
        h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
        h = self.up_block(torch.cat((h, skip), dim=1), condition)
        return self.output_conv(h)


def drop_class_conditions(
    class_labels: torch.Tensor,
    drop_probability: float,
    null_class: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace selected labels with the null class used for unconditional training."""

    if class_labels.ndim != 1:
        raise ValueError("class_labels must have shape (batch,)")
    if class_labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("class_labels must be an integer tensor")
    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError("drop_probability must be between 0 and 1")
    if (
        null_class < 0
        or torch.any(class_labels < 0)
        or torch.any(class_labels >= null_class)
    ):
        raise ValueError("class labels must be between 0 and null_class - 1")
    drop_mask = (
        torch.rand(
            class_labels.shape,
            device=class_labels.device,
            generator=generator,
        )
        < drop_probability
    )
    dropped_labels = torch.where(
        drop_mask, torch.full_like(class_labels, null_class), class_labels
    )
    return dropped_labels, drop_mask


def conditional_noise_prediction_loss(
    model: ConditionalTinyUNet,
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    class_labels: torch.Tensor,
    condition_drop_probability: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute epsilon-prediction MSE with classifier-free condition dropout."""

    if x_start.ndim != 4:
        raise ValueError("x_start must have shape (batch, channels, height, width)")
    if class_labels.shape != (x_start.shape[0],):
        raise ValueError("class_labels must have shape (batch,)")
    timesteps = torch.randint(
        0,
        schedule.num_timesteps,
        (x_start.shape[0],),
        device=x_start.device,
        generator=generator,
    )
    noise = torch.randn(
        x_start.shape,
        device=x_start.device,
        dtype=x_start.dtype,
        generator=generator,
    )
    x_t, noise = q_sample(schedule, x_start, timesteps, noise)
    training_labels, _ = drop_class_conditions(
        class_labels,
        condition_drop_probability,
        model.null_class,
        generator,
    )
    predicted_noise = model(x_t, timesteps, training_labels)
    return F.mse_loss(predicted_noise, noise)


def classifier_free_guidance(
    unconditional_noise: torch.Tensor,
    conditional_noise: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Combine unconditional and conditional epsilon predictions."""

    if unconditional_noise.shape != conditional_noise.shape:
        raise ValueError("noise predictions must have the same shape")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
    return unconditional_noise + guidance_scale * (
        conditional_noise - unconditional_noise
    )


def predict_noise_with_cfg(
    model: ConditionalTinyUNet,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Run unconditional and conditional model passes, then apply CFG."""

    if class_labels.shape != (x_t.shape[0],):
        raise ValueError("class_labels must have shape (batch,)")
    if torch.any(class_labels < 0) or torch.any(class_labels >= model.null_class):
        raise ValueError("class_labels must contain real classes, not the null class")
    null_labels = torch.full_like(class_labels, model.null_class)
    unconditional_noise = model(x_t, timesteps, null_labels)
    conditional_noise = model(x_t, timesteps, class_labels)
    return classifier_free_guidance(
        unconditional_noise, conditional_noise, guidance_scale
    )


def make_ddim_timesteps(num_ddpm_timesteps: int, num_ddim_steps: int) -> torch.Tensor:
    """Return evenly spaced ascending timesteps for DDIM sampling."""

    if num_ddpm_timesteps < 2 or num_ddim_steps < 2:
        raise ValueError("DDPM and DDIM timestep counts must be at least 2")
    if num_ddim_steps > num_ddpm_timesteps:
        raise ValueError("num_ddim_steps cannot exceed num_ddpm_timesteps")
    return torch.linspace(0, num_ddpm_timesteps - 1, num_ddim_steps).long()


def predict_x0_from_noise(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    predicted_noise: torch.Tensor,
) -> torch.Tensor:
    """Estimate x_0 from x_t and predicted epsilon."""

    if predicted_noise.shape != x_t.shape:
        raise ValueError("predicted_noise must have the same shape as x_t")
    sqrt_alpha_bar = gather_by_timestep(
        schedule.sqrt_alpha_bars.to(x_t.device), timesteps, x_t.shape
    )
    sqrt_one_minus_alpha_bar = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars.to(x_t.device), timesteps, x_t.shape
    )
    return (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar


def _previous_alpha_bar(
    schedule: DiffusionSchedule,
    previous_timesteps: torch.Tensor,
    shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    safe_timesteps = previous_timesteps.clamp(min=0)
    alpha_previous = gather_by_timestep(
        schedule.alpha_bars.to(previous_timesteps.device), safe_timesteps, shape
    )
    final_mask = (previous_timesteps < 0).reshape(
        (previous_timesteps.shape[0],) + (1,) * (len(shape) - 1)
    )
    return torch.where(final_mask, torch.ones_like(alpha_previous), alpha_previous)


@torch.no_grad()
def guided_ddim_reverse_step(
    model: ConditionalTinyUNet,
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    previous_timesteps: torch.Tensor,
    class_labels: torch.Tensor,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take one deterministic DDIM step using classifier-free guidance."""

    batch = x_t.shape[0]
    if timesteps.shape != (batch,) or previous_timesteps.shape != (batch,):
        raise ValueError("timesteps must have shape (batch,)")
    if torch.any(previous_timesteps >= timesteps) or torch.any(previous_timesteps < -1):
        raise ValueError("previous timesteps must be in [-1, current timestep)")
    predicted_noise = predict_noise_with_cfg(
        model, x_t, timesteps, class_labels, guidance_scale
    )
    predicted_x0 = predict_x0_from_noise(schedule, x_t, timesteps, predicted_noise)
    alpha_previous = _previous_alpha_bar(
        schedule, previous_timesteps.to(x_t.device), x_t.shape
    )
    x_previous = (
        torch.sqrt(alpha_previous) * predicted_x0
        + torch.sqrt(torch.clamp(1.0 - alpha_previous, min=0.0)) * predicted_noise
    )
    return x_previous, predicted_x0, predicted_noise


@torch.no_grad()
def guided_ddim_sample_loop(
    model: ConditionalTinyUNet,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    class_labels: torch.Tensor,
    guidance_scale: float,
    num_steps: int,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a batch with deterministic guided DDIM sampling."""

    if len(shape) != 4 or shape[0] <= 0:
        raise ValueError("shape must be (batch, channels, height, width)")
    if class_labels.shape != (shape[0],):
        raise ValueError("class_labels must have shape (batch,)")
    device = class_labels.device
    if initial_noise is None:
        x_t = torch.randn(shape, device=device, generator=generator)
    else:
        if initial_noise.shape != shape:
            raise ValueError("initial_noise must match shape")
        x_t = initial_noise.to(device)
    sampling_timesteps = make_ddim_timesteps(schedule.num_timesteps, num_steps).to(
        device
    )
    for index in reversed(range(num_steps)):
        current_value = int(sampling_timesteps[index])
        previous_value = int(sampling_timesteps[index - 1]) if index > 0 else -1
        current = torch.full(
            (shape[0],), current_value, device=device, dtype=torch.long
        )
        previous = torch.full(
            (shape[0],), previous_value, device=device, dtype=torch.long
        )
        x_t, _, _ = guided_ddim_reverse_step(
            model,
            schedule,
            x_t,
            current,
            previous,
            class_labels,
            guidance_scale,
        )
    return x_t
