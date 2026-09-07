"""Provided multimodal encoders, validation, and toy demonstrations for wm.10."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ActionSpec:
    """Per-dimension bounds for continuous robot actions."""

    low: torch.Tensor
    high: torch.Tensor


@dataclass
class VLAPolicyOutput:
    """Normalized and environment-scale action chunk predictions."""

    normalized_actions: torch.Tensor
    actions: torch.Tensor


@dataclass
class VLABatch:
    """A batch of visual histories, instructions, and expert action chunks."""

    videos: torch.Tensor
    instruction_ids: torch.Tensor
    instruction_mask: torch.Tensor
    actions: torch.Tensor
    action_validity: torch.Tensor


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split images into row-major flattened patches."""

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if images.shape[2] != images.shape[3] or images.shape[2] % patch_size != 0:
        raise ValueError("square image size must be divisible by patch_size")
    batch, channels, _, _ = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return (
        patches.permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .reshape(batch, -1, channels * patch_size * patch_size)
    )


class TinyVisualEncoder(nn.Module):
    """Provided position-sensitive patch encoder carried from vision lessons."""

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
    """Provided short-history encoder carried from the video lessons."""

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
    """Provided masked-mean instruction encoder carried from language lessons."""

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


def make_tiny_multimodal_encoders() -> tuple[TinyVideoEncoder, TinyLanguageEncoder]:
    """Return lesson-sized visual/video and language encoders."""

    visual_encoder = TinyVisualEncoder(4, 2, 1, 8, 16)
    video_encoder = TinyVideoEncoder(visual_encoder, frame_count=2, output_dim=16)
    language_encoder = TinyLanguageEncoder(vocab_size=6, embed_dim=8, output_dim=16)
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
    """Return validated action bounds on the action tensor's device and dtype."""

    action_dim = validate_action_spec(action_spec)
    if actions.ndim < 1 or actions.shape[-1] != action_dim:
        raise ValueError("actions last dimension must match the action specification")
    if not torch.is_floating_point(actions) or not torch.isfinite(actions).all():
        raise ValueError("actions must be finite floating-point values")
    low = action_spec.low.to(device=actions.device, dtype=actions.dtype)
    high = action_spec.high.to(device=actions.device, dtype=actions.dtype)
    return low, high


def validate_actions_within_bounds(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> None:
    if (actions < low).any() or (actions > high).any():
        raise ValueError("actions must lie within the action specification")


def validate_normalized_actions(actions: torch.Tensor, action_dim: int) -> None:
    if actions.ndim < 1 or actions.shape[-1] != action_dim:
        raise ValueError("normalized actions have the wrong action dimension")
    if not torch.is_floating_point(actions) or not torch.isfinite(actions).all():
        raise ValueError("normalized actions must be finite floating-point values")
    if (actions < -1.0).any() or (actions > 1.0).any():
        raise ValueError("normalized actions must lie in [-1, 1]")


def validate_policy_configuration(
    video_encoder: TinyVideoEncoder,
    language_encoder: TinyLanguageEncoder,
    action_spec: ActionSpec,
    chunk_size: int,
    hidden_dim: int,
) -> tuple[int, int, int]:
    """Validate policy modules and return their feature/action dimensions."""

    if not isinstance(video_encoder, TinyVideoEncoder):
        raise TypeError("video_encoder must be a TinyVideoEncoder")
    if not isinstance(language_encoder, TinyLanguageEncoder):
        raise TypeError("language_encoder must be a TinyLanguageEncoder")
    if chunk_size <= 0 or hidden_dim <= 0:
        raise ValueError("chunk_size and hidden_dim must be positive")
    action_dim = validate_action_spec(action_spec)
    return video_encoder.output_dim, language_encoder.output_dim, action_dim


def validate_policy_inputs(
    video_encoder: TinyVideoEncoder,
    language_encoder: TinyLanguageEncoder,
    videos: torch.Tensor,
    instruction_ids: torch.Tensor,
    instruction_mask: torch.Tensor,
) -> None:
    """Validate policy inputs before learner fusion code runs."""

    if videos.ndim != 5:
        raise ValueError(
            "videos must have shape [batch, frames, channels, height, width]"
        )
    if instruction_ids.ndim != 2 or instruction_ids.shape[0] != videos.shape[0]:
        raise ValueError("instructions must align with the video batch")
    if instruction_mask.shape != instruction_ids.shape:
        raise ValueError("instruction_mask must match instruction_ids")
    if videos.shape[1] != video_encoder.frame_count:
        raise ValueError("video frame count must match video_encoder")
    if instruction_ids.dtype != torch.long or instruction_mask.dtype != torch.bool:
        raise ValueError("instruction ids must be long and mask must be boolean")
    if (
        instruction_ids.device != videos.device
        or instruction_mask.device != videos.device
    ):
        raise ValueError("policy input tensors must be on the same device")
    if (instruction_ids < 0).any() or (
        instruction_ids >= language_encoder.vocab_size
    ).any():
        raise ValueError("instruction token id is outside the vocabulary")
    if not instruction_mask.any(dim=1).all():
        raise ValueError("every instruction must contain a valid token")


def validate_action_loss_inputs(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
) -> None:
    """Validate masked action regression tensors."""

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


def validate_vla_batch(batch: VLABatch, chunk_size: int, action_dim: int) -> None:
    """Validate demonstration alignment; this is not a learner TODO."""

    if batch.videos.ndim != 5:
        raise ValueError("videos must have five dimensions")
    batch_size = batch.videos.shape[0]
    if batch.instruction_ids.ndim != 2 or batch.instruction_ids.shape[0] != batch_size:
        raise ValueError("instructions must align with videos")
    if batch.instruction_mask.shape != batch.instruction_ids.shape:
        raise ValueError("instruction_mask must match instruction_ids")
    if batch.actions.shape != (batch_size, chunk_size, action_dim):
        raise ValueError("actions must match batch, chunk_size, and action_dim")
    if batch.action_validity.shape != (batch_size, chunk_size):
        raise ValueError("action_validity must match batch and chunk_size")


def _render_positions(positions: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    flat_indices = positions[:, 0] * grid_size + positions[:, 1]
    images = torch.zeros(positions.shape[0], grid_size * grid_size)
    images.scatter_(1, flat_indices.unsqueeze(1), 1.0)
    return images.reshape(positions.shape[0], 1, grid_size, grid_size)


def make_toy_vla_batch() -> VLABatch:
    """Return every position/instruction pair with a two-step expert chunk."""

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
    for _ in range(2):
        next_position = (position + requested_actions.to(torch.long)).clamp(0, 3)
        action_chunks.append((next_position - position).to(torch.float32))
        position = next_position
    actions = torch.stack(action_chunks, dim=1)

    images = _render_positions(current_positions)
    videos = torch.stack((images, images), dim=1)
    instruction_ids = torch.stack(
        (
            torch.ones_like(instruction_directions),
            instruction_directions,
            torch.zeros_like(instruction_directions),
        ),
        dim=1,
    )
    instruction_mask = instruction_ids != 0
    action_validity = torch.ones(actions.shape[:2], dtype=torch.bool)
    return VLABatch(
        videos,
        instruction_ids,
        instruction_mask,
        actions,
        action_validity,
    )
