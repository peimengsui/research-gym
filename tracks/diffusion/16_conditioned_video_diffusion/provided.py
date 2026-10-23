"""Provided conditioning model, diffusion math, and toy motion data."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


PAD_TOKEN_ID = 0
NULL_TOKEN_ID = 1
MOVE_TOKEN_ID = 2
DIRECTION_TOKEN_OFFSET = 3
DIRECTION_NAMES = ("right", "down", "left", "up")
DIRECTION_VECTORS = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
TEXT_VOCABULARY_SIZE = DIRECTION_TOKEN_OFFSET + len(DIRECTION_NAMES)


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed Gaussian diffusion schedule."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


@dataclass(frozen=True)
class ConditionedVideoBatch:
    """Tiny clips paired with first frames and tokenized motion instructions."""

    videos: torch.Tensor
    first_frames: torch.Tensor
    text_token_ids: torch.Tensor
    text_attention_mask: torch.Tensor
    direction_ids: torch.Tensor


@dataclass(frozen=True)
class ConditionedVideoSample:
    """Final conditioned videos and their reverse trajectories."""

    videos: torch.Tensor
    trajectory: torch.Tensor
    sampling_timesteps: torch.Tensor


def make_diffusion_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """Build a linear Gaussian diffusion schedule."""

    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
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
    """Gather one schedule value per batch row and make it broadcastable."""

    if values.ndim != 1 or timesteps.ndim != 1 or timesteps.dtype != torch.long:
        raise ValueError("values must be 1D and timesteps a 1D long tensor")
    if len(broadcast_shape) < 1 or broadcast_shape[0] != timesteps.shape[0]:
        raise ValueError("broadcast_shape must begin with the timestep batch size")
    gathered = values.to(timesteps.device)[timesteps]
    return gathered.reshape((timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1))


def q_sample(
    schedule: DiffusionSchedule,
    clean_videos: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample noisy `[B, C, F, H, W]` videos from clean clips."""

    validate_video_tensor(clean_videos, "clean_videos")
    validate_schedule_timesteps(schedule, timesteps, clean_videos.shape[0])
    if noise is None:
        noise = torch.randn_like(clean_videos)
    if noise.shape != clean_videos.shape:
        raise ValueError("noise must match clean_videos")
    signal = gather_by_timestep(
        schedule.sqrt_alpha_bars,
        timesteps,
        clean_videos.shape,
    ).to(clean_videos)
    noise_scale = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        clean_videos.shape,
    ).to(clean_videos)
    return signal * clean_videos + noise_scale * noise, noise


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Return sinusoidal timestep features with shape `[B, D]`."""

    if timesteps.ndim != 1 or embedding_dim < 2:
        raise ValueError("timesteps must be 1D and embedding_dim at least 2")
    half = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TextContextEncoder(nn.Module):
    """Embed tiny text tokens and their positions."""

    def __init__(self, vocabulary_size: int, max_length: int, context_dim: int) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.max_length = max_length
        self.context_dim = context_dim
        self.token_embedding = nn.Embedding(vocabulary_size, context_dim)
        self.position_embedding = nn.Embedding(max_length, context_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a 2D long tensor")
        if token_ids.shape[1] > self.max_length:
            raise ValueError("text sequence exceeds max_length")
        if token_ids.numel() and (
            token_ids.min() < 0 or token_ids.max() >= self.vocabulary_size
        ):
            raise ValueError("text token ID is outside the vocabulary")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)[None, :, :]
        )


class CrossAttention(nn.Module):
    """Multi-head cross-attention from video features to text context."""

    def __init__(self, feature_dim: int, context_dim: int, num_heads: int) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("num_heads must divide feature_dim")
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(context_dim, feature_dim)
        self.value = nn.Linear(context_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        video_tokens: torch.Tensor,
        text_context: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_queries, _ = video_tokens.shape
        num_text_tokens = text_context.shape[1]
        query = self.query(video_tokens).reshape(
            batch,
            num_queries,
            self.num_heads,
            self.head_dim,
        )
        key = self.key(text_context).reshape(
            batch,
            num_text_tokens,
            self.num_heads,
            self.head_dim,
        )
        value = self.value(text_context).reshape(
            batch,
            num_text_tokens,
            self.num_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~text_attention_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        attended = (
            (weights @ value)
            .transpose(1, 2)
            .reshape(
                batch,
                num_queries,
                self.feature_dim,
            )
        )
        return self.output(attended)


class SpatialDenoisingBlock(nn.Module):
    """Residual 2D processing shared across frames."""

    def __init__(self, channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        self.normalization1 = nn.GroupNorm(1, channels)
        self.convolution1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.normalization2 = nn.GroupNorm(1, channels)
        self.convolution2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        frame_features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.convolution1(F.silu(self.normalization1(frame_features)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.convolution2(F.silu(self.normalization2(hidden)))
        return frame_features + hidden


class TemporalMixingBlock(nn.Module):
    """Residual temporal mixing at each independent spatial location."""

    def __init__(self, channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        self.normalization = nn.GroupNorm(1, channels)
        self.convolution1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.convolution2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(
        self,
        video_features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, frames, height, width = video_features.shape
        sequences = video_features.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width,
            channels,
            frames,
        )
        hidden = self.convolution1(self.normalization(sequences))
        time_bias = self.time_projection(time_embedding)[:, None, None, :, None]
        time_bias = time_bias.expand(batch, height, width, channels, 1)
        hidden = F.silu(hidden + time_bias.reshape(batch * height * width, channels, 1))
        output = sequences + self.convolution2(hidden)
        return output.reshape(
            batch,
            height,
            width,
            channels,
            frames,
        ).permute(0, 3, 4, 1, 2)


class ConditionedVideoDenoiser(nn.Module):
    """Spatiotemporal epsilon predictor with first-frame and text context."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        condition_dim: int = 32,
        num_attention_heads: int = 4,
        max_text_length: int = 2,
    ) -> None:
        super().__init__()
        if hidden_channels % num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_channels")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.condition_dim = condition_dim
        self.null_token_id = NULL_TOKEN_ID
        self.time_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim),
        )
        self.text_encoder = TextContextEncoder(
            TEXT_VOCABULARY_SIZE,
            max_text_length,
            condition_dim,
        )
        self.text_condition_projection = nn.Linear(condition_dim, condition_dim)
        self.input_projection = nn.Conv2d(
            in_channels * 2,
            hidden_channels,
            3,
            padding=1,
        )
        self.spatial_block1 = SpatialDenoisingBlock(hidden_channels, condition_dim)
        self.temporal_block = TemporalMixingBlock(hidden_channels, condition_dim)
        self.token_normalization = nn.LayerNorm(hidden_channels)
        self.cross_attention = CrossAttention(
            hidden_channels,
            condition_dim,
            num_attention_heads,
        )
        self.spatial_block2 = SpatialDenoisingBlock(hidden_channels, condition_dim)
        self.output_projection = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(
        self,
        noisy_videos: torch.Tensor,
        timesteps: torch.Tensor,
        first_frames: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        validate_conditioned_model_inputs(
            noisy_videos,
            timesteps,
            first_frames,
            text_token_ids,
            text_attention_mask,
            self.in_channels,
        )
        batch, _, frames, height, width = noisy_videos.shape
        text_context = self.text_encoder(text_token_ids)
        text_weights = text_attention_mask.to(text_context.dtype)[..., None]
        pooled_text = (text_context * text_weights).sum(dim=1)
        pooled_text = pooled_text / text_weights.sum(dim=1).clamp_min(1.0)
        time_embedding = self.time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.condition_dim).to(
                noisy_videos
            )
        )
        # A pooled text bias makes the tiny CPU model's global motion command
        # easy to learn; cross-attention still supplies token-level context.
        time_embedding = time_embedding + self.text_condition_projection(pooled_text)
        frame_time = time_embedding[:, None, :].expand(
            batch,
            frames,
            self.condition_dim,
        )
        frame_time = frame_time.reshape(batch * frames, self.condition_dim)

        repeated_first_frame = first_frames[:, :, None].expand(
            batch,
            self.in_channels,
            frames,
            height,
            width,
        )
        model_input = torch.cat((noisy_videos, repeated_first_frame), dim=1)
        frame_features = _video_to_frame_batch(model_input)
        frame_features = F.silu(self.input_projection(frame_features))
        frame_features = self.spatial_block1(frame_features, frame_time)
        video_features = _frame_batch_to_video(frame_features, batch, frames)
        video_features = self.temporal_block(video_features, time_embedding)

        video_tokens = video_features.permute(0, 2, 3, 4, 1).reshape(
            batch,
            frames * height * width,
            self.hidden_channels,
        )
        attended = self.cross_attention(
            self.token_normalization(video_tokens),
            text_context,
            text_attention_mask,
        )
        video_tokens = video_tokens + attended
        video_features = video_tokens.reshape(
            batch,
            frames,
            height,
            width,
            self.hidden_channels,
        ).permute(0, 4, 1, 2, 3)

        frame_features = _video_to_frame_batch(video_features)
        frame_features = self.spatial_block2(frame_features, frame_time)
        predicted_noise = self.output_projection(frame_features)
        return _frame_batch_to_video(predicted_noise, batch, frames)


def drop_text_conditions(
    text_token_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    drop_probability: float,
    null_token_id: int = NULL_TOKEN_ID,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace every valid text token in selected rows with the null token."""

    validate_text_inputs(text_token_ids, text_attention_mask)
    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError("drop_probability must lie in [0, 1]")
    drop_mask = (
        torch.rand(
            (text_token_ids.shape[0],),
            device=text_token_ids.device,
            generator=generator,
        )
        < drop_probability
    )
    replaced = torch.where(
        drop_mask[:, None] & text_attention_mask,
        torch.full_like(text_token_ids, null_token_id),
        text_token_ids,
    )
    return replaced, drop_mask


def conditioned_video_noise_prediction_loss(
    model: ConditionedVideoDenoiser,
    schedule: DiffusionSchedule,
    clean_videos: torch.Tensor,
    text_token_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    condition_drop_probability: float = 0.1,
    foreground_weight: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Train future-frame epsilon prediction with classifier-free text dropout."""

    validate_video_tensor(clean_videos, "clean_videos")
    validate_text_inputs(text_token_ids, text_attention_mask, clean_videos.shape[0])
    if foreground_weight < 0:
        raise ValueError("foreground_weight must be non-negative")
    timesteps = torch.randint(
        0,
        schedule.num_timesteps,
        (clean_videos.shape[0],),
        device=clean_videos.device,
        generator=generator,
    )
    noise = torch.randn(
        clean_videos.shape,
        device=clean_videos.device,
        dtype=clean_videos.dtype,
        generator=generator,
    )
    noisy_videos, noise = q_sample(schedule, clean_videos, timesteps, noise)
    training_tokens, _ = drop_text_conditions(
        text_token_ids,
        text_attention_mask,
        condition_drop_probability,
        model.null_token_id,
        generator,
    )
    predicted_noise = model(
        noisy_videos,
        timesteps,
        clean_videos[:, :, 0],
        training_tokens,
        text_attention_mask,
    )
    squared_error = (predicted_noise[:, :, 1:] - noise[:, :, 1:]).square()
    if foreground_weight == 0.0:
        return squared_error.mean()
    foreground = ((clean_videos[:, :, 1:] + 1.0) / 2.0).clamp(0.0, 1.0)
    weights = 1.0 + foreground_weight * foreground
    return (weights * squared_error).sum() / weights.sum()


def classifier_free_guidance(
    unconditional_noise: torch.Tensor,
    conditional_noise: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Combine text-unconditional and text-conditional epsilon predictions."""

    if unconditional_noise.shape != conditional_noise.shape:
        raise ValueError("noise predictions must have the same shape")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
    return unconditional_noise + guidance_scale * (
        conditional_noise - unconditional_noise
    )


def predict_noise_with_cfg(
    model: ConditionedVideoDenoiser,
    noisy_videos: torch.Tensor,
    timesteps: torch.Tensor,
    first_frames: torch.Tensor,
    text_token_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Predict guided noise while retaining the first-frame condition in both passes."""

    null_tokens = torch.where(
        text_attention_mask,
        torch.full_like(text_token_ids, model.null_token_id),
        text_token_ids,
    )
    unconditional_noise = model(
        noisy_videos,
        timesteps,
        first_frames,
        null_tokens,
        text_attention_mask,
    )
    conditional_noise = model(
        noisy_videos,
        timesteps,
        first_frames,
        text_token_ids,
        text_attention_mask,
    )
    return classifier_free_guidance(
        unconditional_noise,
        conditional_noise,
        guidance_scale,
    )


def make_ddim_timesteps(
    num_training_steps: int, num_sampling_steps: int
) -> torch.Tensor:
    """Return ascending selected timesteps for a shortened DDIM trajectory."""

    if num_training_steps < 2 or num_sampling_steps < 2:
        raise ValueError("training and sampling schedules need at least 2 steps")
    if num_sampling_steps > num_training_steps:
        raise ValueError("num_sampling_steps cannot exceed num_training_steps")
    return torch.linspace(0, num_training_steps - 1, num_sampling_steps).long()


@torch.no_grad()
def ddim_step_from_noise(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    previous_timesteps: torch.Tensor,
    predicted_noise: torch.Tensor,
    clip_denoised: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take one deterministic DDIM update from an already-guided prediction."""

    validate_video_tensor(x_t, "x_t")
    validate_schedule_timesteps(schedule, timesteps, x_t.shape[0])
    if previous_timesteps.shape != timesteps.shape:
        raise ValueError("previous_timesteps must match timesteps")
    if (previous_timesteps >= timesteps).any() or (previous_timesteps < -1).any():
        raise ValueError("previous timesteps must be smaller and no less than -1")
    if predicted_noise.shape != x_t.shape:
        raise ValueError("predicted_noise must match x_t")

    alpha_t = gather_by_timestep(schedule.alpha_bars, timesteps, x_t.shape).to(x_t)
    predicted_x0 = (x_t - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(
        alpha_t
    )
    if clip_denoised:
        predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

    safe_previous = previous_timesteps.clamp(min=0)
    alpha_previous = gather_by_timestep(
        schedule.alpha_bars,
        safe_previous,
        x_t.shape,
    ).to(x_t)
    final_mask = (previous_timesteps < 0).reshape(
        (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
    )
    alpha_previous = torch.where(
        final_mask,
        torch.ones_like(alpha_previous),
        alpha_previous,
    )
    x_previous = (
        torch.sqrt(alpha_previous) * predicted_x0
        + torch.sqrt(1.0 - alpha_previous) * predicted_noise
    )
    return x_previous, predicted_x0


def frame_centers(videos: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Return soft foreground centers with shape `[B, F, 2]`."""

    validate_video_tensor(videos)
    _, _, _, height, width = videos.shape
    foreground = ((videos + 1.0) / 2.0).clamp(0.0, 1.0).mean(dim=1)
    mass = foreground.sum(dim=(-2, -1))
    rows = torch.arange(height, device=videos.device, dtype=videos.dtype).reshape(
        1, 1, height, 1
    )
    columns = torch.arange(width, device=videos.device, dtype=videos.dtype).reshape(
        1, 1, 1, width
    )
    centers = torch.stack(
        (
            (foreground * rows).sum(dim=(-2, -1)) / mass.clamp_min(epsilon),
            (foreground * columns).sum(dim=(-2, -1)) / mass.clamp_min(epsilon),
        ),
        dim=-1,
    )
    image_center = videos.new_tensor(((height - 1) / 2, (width - 1) / 2))
    return torch.where((mass <= epsilon)[..., None], image_center, centers)


def make_conditioned_moving_dot_batch(
    batch_size: int = 8,
    num_frames: int = 4,
    image_size: int = 8,
) -> ConditionedVideoBatch:
    """Create clips sharing a first frame but following four text directions."""

    if batch_size <= 0 or num_frames < 3 or image_size < 2 * num_frames:
        raise ValueError("use a positive batch, at least 3 frames, and enough space")
    videos = torch.full(
        (batch_size, 1, num_frames, image_size, image_size),
        -1.0,
    )
    direction_ids = torch.arange(batch_size, dtype=torch.long) % len(DIRECTION_NAMES)
    start = num_frames - 1
    for sample in range(batch_size):
        delta_row, delta_column = DIRECTION_VECTORS[direction_ids[sample]].long()
        for frame in range(num_frames):
            row = start + frame * int(delta_row.item())
            column = start + frame * int(delta_column.item())
            videos[sample, 0, frame, row : row + 2, column : column + 2] = 1.0
    text_token_ids = torch.stack(
        (
            torch.full((batch_size,), MOVE_TOKEN_ID, dtype=torch.long),
            DIRECTION_TOKEN_OFFSET + direction_ids,
        ),
        dim=1,
    )
    return ConditionedVideoBatch(
        videos=videos,
        first_frames=videos[:, :, 0].clone(),
        text_token_ids=text_token_ids,
        text_attention_mask=torch.ones_like(text_token_ids, dtype=torch.bool),
        direction_ids=direction_ids,
    )


def validate_video_tensor(videos: torch.Tensor, name: str = "videos") -> None:
    """Validate floating-point `[B, C, F, H, W]` video tensors."""

    if videos.ndim != 5 or not videos.is_floating_point():
        raise ValueError(f"{name} must be floating point with shape (B, C, F, H, W)")
    if any(size <= 0 for size in videos.shape):
        raise ValueError(f"{name} dimensions must be positive")


def validate_first_frames(
    first_frames: torch.Tensor, name: str = "first_frames"
) -> None:
    """Validate floating-point `[B, C, H, W]` first frames."""

    if first_frames.ndim != 4 or not first_frames.is_floating_point():
        raise ValueError(f"{name} must be floating point with shape (B, C, H, W)")
    if any(size <= 0 for size in first_frames.shape):
        raise ValueError(f"{name} dimensions must be positive")


def validate_text_inputs(
    text_token_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    batch_size: int | None = None,
) -> None:
    """Validate token IDs and a nonempty boolean attention mask."""

    if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
        raise ValueError("text_token_ids must be a 2D long tensor")
    if text_attention_mask.shape != text_token_ids.shape:
        raise ValueError("text_attention_mask must match text_token_ids")
    if (
        text_attention_mask.dtype != torch.bool
        or not text_attention_mask.any(dim=1).all()
    ):
        raise ValueError("each row needs at least one valid text token")
    if batch_size is not None and text_token_ids.shape[0] != batch_size:
        raise ValueError("text batch size does not match the video batch")


def validate_schedule_timesteps(
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor,
    batch_size: int,
) -> None:
    """Validate timestep shape, dtype, and schedule range."""

    if timesteps.shape != (batch_size,) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long with shape (batch,)")
    if timesteps.numel() and (
        timesteps.min() < 0 or timesteps.max() >= schedule.num_timesteps
    ):
        raise ValueError("timesteps are outside the diffusion schedule")


def validate_known_frame_inputs(
    schedule: DiffusionSchedule,
    first_frames: torch.Tensor,
    timesteps: torch.Tensor,
    frame_noise: torch.Tensor,
) -> None:
    """Validate inputs for a known first-frame diffusion trajectory."""

    validate_first_frames(first_frames)
    if frame_noise.shape != first_frames.shape:
        raise ValueError("frame_noise must match first_frames")
    if timesteps.shape != (first_frames.shape[0],) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long with shape (batch,)")
    if timesteps.numel() and (
        timesteps.min() < -1 or timesteps.max() >= schedule.num_timesteps
    ):
        raise ValueError("known-frame timesteps must lie in [-1, T-1]")


def validate_first_frame_constraint_inputs(
    videos: torch.Tensor,
    constrained_first_frames: torch.Tensor,
) -> None:
    """Validate a video and replacement frame before hard clamping."""

    validate_video_tensor(videos)
    validate_first_frames(constrained_first_frames, "constrained_first_frames")
    if constrained_first_frames.shape != (
        videos.shape[0],
        videos.shape[1],
        videos.shape[3],
        videos.shape[4],
    ):
        raise ValueError("constrained first frames do not match the video")


def validate_conditioned_model_inputs(
    noisy_videos: torch.Tensor,
    timesteps: torch.Tensor,
    first_frames: torch.Tensor,
    text_token_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    in_channels: int,
) -> None:
    """Validate the conditioned denoiser's public input contract."""

    validate_video_tensor(noisy_videos, "noisy_videos")
    validate_first_frames(first_frames)
    validate_text_inputs(text_token_ids, text_attention_mask, noisy_videos.shape[0])
    if noisy_videos.shape[1] != in_channels:
        raise ValueError("video channels do not match the model")
    if first_frames.shape != (
        noisy_videos.shape[0],
        noisy_videos.shape[1],
        noisy_videos.shape[3],
        noisy_videos.shape[4],
    ):
        raise ValueError("first_frames must match video batch, channels, and space")
    if timesteps.shape != (noisy_videos.shape[0],) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long with shape (batch,)")


def _video_to_frame_batch(videos: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = videos.shape
    return videos.permute(0, 2, 1, 3, 4).reshape(
        batch * frames,
        channels,
        height,
        width,
    )


def _frame_batch_to_video(
    frames: torch.Tensor,
    batch_size: int,
    num_frames: int,
) -> torch.Tensor:
    _, channels, height, width = frames.shape
    return frames.reshape(
        batch_size,
        num_frames,
        channels,
        height,
        width,
    ).permute(0, 2, 1, 3, 4)
