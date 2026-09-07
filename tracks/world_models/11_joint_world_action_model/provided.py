"""Provided encoders, data, validation, and carried-over VLA utilities for wm.11."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActionSpec:
    """Per-dimension bounds for continuous actions."""

    low: torch.Tensor
    high: torch.Tensor


@dataclass
class WAMBatch:
    """Aligned observations, instructions, actions, and next observations."""

    current_videos: torch.Tensor
    instruction_ids: torch.Tensor
    instruction_mask: torch.Tensor
    action_chunks: torch.Tensor
    action_validity: torch.Tensor
    transition_actions: torch.Tensor
    next_videos: torch.Tensor


@dataclass
class JointWAMOutput:
    """Action predictions and one-step latent transition predictions."""

    state_latents: torch.Tensor
    normalized_actions: torch.Tensor
    actions: torch.Tensor
    predicted_next_latents: torch.Tensor
    target_next_latents: torch.Tensor


@dataclass
class JointWAMLosses:
    """Total objective and its two inspectable components."""

    total_loss: torch.Tensor
    action_loss: torch.Tensor
    latent_loss: torch.Tensor


@dataclass(frozen=True)
class JointWAMMetrics:
    """Detached scalar metrics from one optimizer step."""

    total_loss: float
    action_loss: float
    latent_loss: float


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split images into row-major flattened patches."""

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if images.shape[2] != images.shape[3] or images.shape[2] % patch_size != 0:
        raise ValueError("square image size must be divisible by patch_size")
    batch, channels, _, _ = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(
        3,
        patch_size,
        patch_size,
    )
    return (
        patches.permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .reshape(batch, -1, channels * patch_size * patch_size)
    )


class TinyVisualEncoder(nn.Module):
    """Provided position-sensitive patch encoder carried from earlier lessons."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        patch_dim: int,
        output_dim: int,
    ):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("patch_size must positively divide image_size")
        if min(in_channels, patch_dim, output_dim) <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.output_dim = output_dim
        self.patch_projection = nn.Linear(
            in_channels * patch_size * patch_size,
            patch_dim,
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, patch_dim) * 0.02
        )
        self.output = nn.Sequential(
            nn.Linear(self.num_patches * patch_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(
                "images must have shape "
                f"[batch, {self.in_channels}, {self.image_size}, {self.image_size}]"
            )
        patch_tokens = self.patch_projection(patchify(images, self.patch_size))
        patch_tokens = patch_tokens + self.position_embedding
        return self.output(patch_tokens.flatten(start_dim=1))


class TinyVideoEncoder(nn.Module):
    """Provided short-history encoder carried from the VLA lesson."""

    def __init__(
        self,
        visual_encoder: TinyVisualEncoder,
        frame_count: int,
        output_dim: int,
    ):
        super().__init__()
        if frame_count <= 0 or output_dim <= 0:
            raise ValueError("frame_count and output_dim must be positive")
        self.visual_encoder = visual_encoder
        self.frame_count = frame_count
        self.output_dim = output_dim
        self.temporal_position_embedding = nn.Parameter(
            torch.randn(1, frame_count, visual_encoder.output_dim) * 0.02
        )
        self.temporal_mixer = nn.Sequential(
            nn.Linear(frame_count * visual_encoder.output_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        visual = self.visual_encoder
        expected = (
            self.frame_count,
            visual.in_channels,
            visual.image_size,
            visual.image_size,
        )
        if videos.ndim != 5 or videos.shape[1:] != expected:
            raise ValueError(
                "videos must have shape "
                "[batch, frame_count, channels, image_size, image_size]"
            )
        batch = videos.shape[0]
        frames = videos.reshape(batch * self.frame_count, *videos.shape[2:])
        frame_features = self.visual_encoder(frames).reshape(
            batch,
            self.frame_count,
            -1,
        )
        frame_features = frame_features + self.temporal_position_embedding
        return self.temporal_mixer(frame_features.flatten(start_dim=1))


class TinyLanguageEncoder(nn.Module):
    """Provided masked-mean instruction encoder carried from the VLA lesson."""

    def __init__(self, vocab_size: int, embed_dim: int, output_dim: int):
        super().__init__()
        if min(vocab_size, embed_dim, output_dim) <= 0:
            raise ValueError("language dimensions must be positive")
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        if instruction_ids.ndim != 2 or instruction_ids.dtype != torch.long:
            raise ValueError("instruction_ids must be a long [batch, tokens] tensor")
        if (
            instruction_mask.shape != instruction_ids.shape
            or instruction_mask.dtype != torch.bool
        ):
            raise ValueError(
                "instruction_mask must be boolean and match instruction_ids"
            )
        if not instruction_mask.any(dim=1).all():
            raise ValueError("every instruction must contain a valid token")
        if (instruction_ids < 0).any() or (instruction_ids >= self.vocab_size).any():
            raise ValueError("instruction token id is outside the vocabulary")
        token_embeddings = self.token_embedding(instruction_ids)
        weights = instruction_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1)
        return self.output(pooled)


def make_frozen_multimodal_encoders(
    seed: int = 11,
) -> tuple[TinyVideoEncoder, TinyLanguageEncoder]:
    """Return deterministic frozen encoders carried from wm.10."""

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        visual_encoder = TinyVisualEncoder(4, 2, 1, 8, 16)
        video_encoder = TinyVideoEncoder(visual_encoder, frame_count=2, output_dim=16)
        language_encoder = TinyLanguageEncoder(
            vocab_size=6,
            embed_dim=8,
            output_dim=16,
        )
    video_encoder.requires_grad_(False)
    language_encoder.requires_grad_(False)
    video_encoder.eval()
    language_encoder.eval()
    return video_encoder, language_encoder


def validate_action_spec(action_spec: ActionSpec) -> int:
    """Validate action bounds and return action dimension."""

    if action_spec.low.ndim != 1 or action_spec.high.shape != action_spec.low.shape:
        raise ValueError("action bounds must be matching one-dimensional tensors")
    if action_spec.low.numel() == 0:
        raise ValueError("action bounds must be non-empty")
    if (
        not torch.isfinite(action_spec.low).all()
        or not torch.isfinite(action_spec.high).all()
    ):
        raise ValueError("action bounds must be finite")
    if not (action_spec.high > action_spec.low).all():
        raise ValueError("every high action bound must exceed low")
    return action_spec.low.numel()


def action_bounds_for(
    actions: torch.Tensor,
    action_spec: ActionSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return action bounds on the action tensor's device and dtype."""

    action_dim = validate_action_spec(action_spec)
    if actions.ndim < 1 or actions.shape[-1] != action_dim:
        raise ValueError("actions last dimension must match the action specification")
    if not torch.is_floating_point(actions) or not torch.isfinite(actions).all():
        raise ValueError("actions must be finite floating-point values")
    low = action_spec.low.to(device=actions.device, dtype=actions.dtype)
    high = action_spec.high.to(device=actions.device, dtype=actions.dtype)
    return low, high


def normalize_actions(actions: torch.Tensor, action_spec: ActionSpec) -> torch.Tensor:
    """Carried from wm.10: map bounded actions to [-1, 1]."""

    low, high = action_bounds_for(actions, action_spec)
    if (actions < low).any() or (actions > high).any():
        raise ValueError("actions must lie within the action specification")
    return 2.0 * (actions - low) / (high - low) - 1.0


def denormalize_actions(
    normalized_actions: torch.Tensor,
    action_spec: ActionSpec,
) -> torch.Tensor:
    """Carried from wm.10: map normalized actions to environment scale."""

    low, high = action_bounds_for(normalized_actions, action_spec)
    if (normalized_actions < -1.0).any() or (normalized_actions > 1.0).any():
        raise ValueError("normalized actions must lie in [-1, 1]")
    return low + 0.5 * (normalized_actions + 1.0) * (high - low)


def masked_action_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Carried from wm.10: average action error over valid elements."""

    if predictions.ndim != 3 or targets.shape != predictions.shape:
        raise ValueError(
            "predictions and targets must share [batch, chunk, action_dim]"
        )
    if validity.shape != predictions.shape[:2] or validity.dtype != torch.bool:
        raise ValueError("validity must be boolean [batch, chunk]")
    if not validity.any():
        raise ValueError("action batch must contain at least one valid step")
    if targets.device != predictions.device or validity.device != predictions.device:
        raise ValueError("action loss tensors must be on the same device")
    expanded_validity = validity.unsqueeze(-1).expand_as(predictions)
    return ((predictions - targets) ** 2).masked_select(expanded_validity).mean()


def validate_model_configuration(
    video_encoder: TinyVideoEncoder,
    language_encoder: TinyLanguageEncoder,
    action_spec: ActionSpec,
    chunk_size: int,
    state_dim: int,
    hidden_dim: int,
) -> tuple[int, int, int]:
    """Validate model modules and return video, language, and action dimensions."""

    if not isinstance(video_encoder, TinyVideoEncoder):
        raise TypeError("video_encoder must be a TinyVideoEncoder")
    if not isinstance(language_encoder, TinyLanguageEncoder):
        raise TypeError("language_encoder must be a TinyLanguageEncoder")
    if min(chunk_size, state_dim, hidden_dim) <= 0:
        raise ValueError("chunk_size, state_dim, and hidden_dim must be positive")
    action_dim = validate_action_spec(action_spec)
    return video_encoder.output_dim, language_encoder.output_dim, action_dim


def validate_observation_inputs(
    video_encoder: TinyVideoEncoder,
    language_encoder: TinyLanguageEncoder,
    current_videos: torch.Tensor,
    instruction_ids: torch.Tensor,
    instruction_mask: torch.Tensor,
    next_videos: torch.Tensor,
) -> None:
    """Validate observations and instructions before learner code runs."""

    visual = video_encoder.visual_encoder
    expected_video_shape = (
        video_encoder.frame_count,
        visual.in_channels,
        visual.image_size,
        visual.image_size,
    )
    if current_videos.ndim != 5 or current_videos.shape[1:] != expected_video_shape:
        raise ValueError("current_videos do not match the video encoder")
    if next_videos.shape != current_videos.shape:
        raise ValueError("next_videos must match current_videos")
    if instruction_ids.ndim != 2 or instruction_ids.shape[0] != current_videos.shape[0]:
        raise ValueError("instructions must align with the video batch")
    if instruction_mask.shape != instruction_ids.shape:
        raise ValueError("instruction_mask must match instruction_ids")
    if instruction_ids.dtype != torch.long or instruction_mask.dtype != torch.bool:
        raise ValueError("instruction ids must be long and mask must be boolean")
    if not instruction_mask.any(dim=1).all():
        raise ValueError("every instruction must contain a valid token")
    if (instruction_ids < 0).any() or (
        instruction_ids >= language_encoder.vocab_size
    ).any():
        raise ValueError("instruction token id is outside the vocabulary")
    tensors = (instruction_ids, instruction_mask, next_videos)
    if any(tensor.device != current_videos.device for tensor in tensors):
        raise ValueError("observation and instruction tensors must share a device")


def validate_transition_inputs(
    state_latents: torch.Tensor,
    transition_actions: torch.Tensor,
    state_dim: int,
    action_spec: ActionSpec,
) -> None:
    """Validate latent dynamics inputs."""

    action_dim = validate_action_spec(action_spec)
    if state_latents.ndim != 2 or state_latents.shape[1] != state_dim:
        raise ValueError("state_latents must have shape [batch, state_dim]")
    if transition_actions.shape != (state_latents.shape[0], action_dim):
        raise ValueError("transition_actions must have shape [batch, action_dim]")
    if transition_actions.device != state_latents.device:
        raise ValueError("state latents and transition actions must share a device")


def validate_wam_batch(
    batch: WAMBatch,
    video_encoder: TinyVideoEncoder,
    language_encoder: TinyLanguageEncoder,
    action_spec: ActionSpec,
    chunk_size: int,
) -> None:
    """Validate aligned demonstrations; this is not a learner TODO."""

    validate_observation_inputs(
        video_encoder,
        language_encoder,
        batch.current_videos,
        batch.instruction_ids,
        batch.instruction_mask,
        batch.next_videos,
    )
    batch_size = batch.current_videos.shape[0]
    action_dim = validate_action_spec(action_spec)
    expected_chunk_shape = (batch_size, chunk_size, action_dim)
    if batch.action_chunks.shape != expected_chunk_shape:
        raise ValueError("action_chunks must match batch, chunk_size, and action_dim")
    if batch.action_validity.shape != (batch_size, chunk_size):
        raise ValueError("action_validity must match batch and chunk_size")
    if batch.action_validity.dtype != torch.bool:
        raise ValueError("action_validity must be boolean")
    if batch.transition_actions.shape != (batch_size, action_dim):
        raise ValueError("transition_actions must match batch and action_dim")
    action_tensors = (
        batch.action_chunks,
        batch.action_validity,
        batch.transition_actions,
    )
    if any(tensor.device != batch.current_videos.device for tensor in action_tensors):
        raise ValueError("all batch tensors must share a device")
    normalize_actions(batch.action_chunks, action_spec)
    normalize_actions(batch.transition_actions, action_spec)
    if not torch.allclose(batch.transition_actions, batch.action_chunks[:, 0]):
        raise ValueError("transition_actions must equal the first action in each chunk")


def validate_loss_weights(action_weight: float, latent_weight: float) -> None:
    """Require a meaningful nonnegative joint objective."""

    if action_weight < 0.0 or latent_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if action_weight == 0.0 and latent_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")


def _render_positions(positions: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    flat_indices = positions[:, 0] * grid_size + positions[:, 1]
    images = torch.zeros(positions.shape[0], grid_size * grid_size)
    images.scatter_(1, flat_indices.unsqueeze(1), 1.0)
    return images.reshape(positions.shape[0], 1, grid_size, grid_size)


def make_toy_wam_batch() -> WAMBatch:
    """Return grid demonstrations with actions and their one-step consequences."""

    # Vocabulary: 0=<pad>, 1=move, 2=up, 3=down, 4=left, 5=right.
    direction_ids = torch.tensor([2, 3, 4, 5])
    deltas = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    coordinates = torch.arange(4)
    positions = torch.cartesian_prod(coordinates, coordinates)
    current_positions = positions.repeat_interleave(4, dim=0)
    instruction_directions = direction_ids.repeat(positions.shape[0])
    requested_actions = deltas.repeat(positions.shape[0], 1)

    action_chunks = []
    position = current_positions.clone()
    next_positions = None
    for step in range(2):
        updated_position = (position + requested_actions.to(torch.long)).clamp(0, 3)
        action_chunks.append((updated_position - position).to(torch.float32))
        position = updated_position
        if step == 0:
            next_positions = position.clone()
    actions = torch.stack(action_chunks, dim=1)
    if next_positions is None:
        raise RuntimeError("toy action chunk must contain a transition")

    current_images = _render_positions(current_positions)
    next_images = _render_positions(next_positions)
    current_videos = torch.stack((current_images, current_images), dim=1)
    next_videos = torch.stack((next_images, next_images), dim=1)
    instruction_ids = torch.stack(
        (
            torch.ones_like(instruction_directions),
            instruction_directions,
            torch.zeros_like(instruction_directions),
        ),
        dim=1,
    )
    return WAMBatch(
        current_videos=current_videos,
        instruction_ids=instruction_ids,
        instruction_mask=instruction_ids != 0,
        action_chunks=actions,
        action_validity=torch.ones(actions.shape[:2], dtype=torch.bool),
        transition_actions=actions[:, 0],
        next_videos=next_videos,
    )
