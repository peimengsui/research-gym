"""Frozen JEPA encoder, validation, and grid-world data for wm.07."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ActionConditionedOutput:
    """One-step latent transition predictions and their training loss."""

    predicted_next_latents: torch.Tensor
    target_next_latents: torch.Tensor
    loss: torch.Tensor


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split square images into row-major flattened patches.

    images: [batch, channels, image_size, image_size]
    returns: [batch, num_patches, channels * patch_size * patch_size]
    """

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if images.shape[2] != images.shape[3]:
        raise ValueError("images must be square")
    if images.shape[2] % patch_size != 0:
        raise ValueError("image size must be divisible by patch_size")

    batch, channels, _, _ = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return (
        patches.permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .reshape(batch, -1, channels * patch_size * patch_size)
    )


class TinyPatchEncoder(nn.Module):
    """Completed patch-level JEPA representation encoder from wm.06."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("patch_size must positively divide image_size")
        if in_channels <= 0 or embed_dim <= 0:
            raise ValueError("in_channels and embed_dim must be positive")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected_shape = (
            self.in_channels,
            self.image_size,
            self.image_size,
        )
        if images.ndim != 4 or images.shape[1:] != expected_shape:
            raise ValueError(
                "images must have shape "
                f"[batch, {self.in_channels}, {self.image_size}, {self.image_size}]"
            )
        patches = patchify(images, self.patch_size)
        return self.normalization(self.projection(patches) + self.position_embedding)


def make_frozen_jepa_encoder(
    image_size: int = 4,
    patch_size: int = 2,
    in_channels: int = 1,
    embed_dim: int = 8,
    seed: int = 6,
) -> TinyPatchEncoder:
    """Return a deterministic frozen encoder standing in for wm.06 training."""

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        encoder = TinyPatchEncoder(
            image_size,
            patch_size,
            in_channels,
            embed_dim,
        )
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def grid_action_set() -> torch.Tensor:
    """Return stay, up, down, left, and right as [dy, dx]."""

    return torch.tensor(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )


def render_grid_positions(
    positions: torch.Tensor,
    grid_size: int = 4,
) -> torch.Tensor:
    """Render integer [row, column] positions as one-hot images."""

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [batch, 2]")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must use an integer dtype")
    if (positions < 0).any() or (positions >= grid_size).any():
        raise ValueError("positions must lie inside the grid")

    flat_indices = positions[:, 0] * grid_size + positions[:, 1]
    images = torch.zeros(
        positions.shape[0],
        grid_size * grid_size,
        device=positions.device,
    )
    images.scatter_(1, flat_indices.unsqueeze(1), 1.0)
    return images.reshape(positions.shape[0], 1, grid_size, grid_size)


def make_grid_transition_batch(
    grid_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return every state-action transition in a bounded moving-dot world."""

    if grid_size <= 1:
        raise ValueError("grid_size must be at least two")
    coordinates = torch.arange(grid_size)
    positions = torch.cartesian_prod(coordinates, coordinates)
    action_set = grid_action_set()
    current_positions = positions.repeat_interleave(action_set.shape[0], dim=0)
    actions = action_set.repeat(positions.shape[0], 1)
    next_positions = current_positions + actions.to(torch.long)
    next_positions = next_positions.clamp(0, grid_size - 1)
    return (
        render_grid_positions(current_positions, grid_size),
        actions,
        render_grid_positions(next_positions, grid_size),
    )


def enumerate_action_sequences(
    horizon: int,
    action_set: torch.Tensor | None = None,
) -> torch.Tensor:
    """Enumerate short candidate sequences as [candidates, horizon, action_dim]."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if action_set is None:
        action_set = grid_action_set()
    if action_set.ndim != 2 or action_set.shape[0] == 0:
        raise ValueError("action_set must have shape [num_actions, action_dim]")

    axes = [torch.arange(action_set.shape[0]) for _ in range(horizon)]
    action_indices = torch.cartesian_prod(*axes)
    if horizon == 1:
        action_indices = action_indices.unsqueeze(1)
    return action_set[action_indices]


def apply_grid_actions(
    initial_position: torch.Tensor,
    actions: torch.Tensor,
    grid_size: int = 4,
) -> torch.Tensor:
    """Apply [dy, dx] actions and return positions for all rollout steps."""

    if initial_position.shape != (2,) or initial_position.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("initial_position must be an integer [2] tensor")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [horizon, 2]")
    if grid_size <= 1:
        raise ValueError("grid_size must be at least two")

    current = initial_position.clone()
    positions = [current.clone()]
    for action in actions:
        current = (current + action.to(torch.long)).clamp(0, grid_size - 1)
        positions.append(current.clone())
    return torch.stack(positions)


def validate_model_configuration(
    encoder: TinyPatchEncoder,
    action_dim: int,
    hidden_dim: int,
) -> int:
    """Validate model sizes and return the flattened latent width."""

    if not isinstance(encoder, TinyPatchEncoder):
        raise TypeError("encoder must be a TinyPatchEncoder")
    if action_dim <= 0 or hidden_dim <= 0:
        raise ValueError("action_dim and hidden_dim must be positive")
    return encoder.num_patches * encoder.embed_dim


def validate_transition_batch(
    encoder: TinyPatchEncoder,
    current_images: torch.Tensor,
    actions: torch.Tensor,
    next_images: torch.Tensor,
    action_dim: int,
) -> None:
    """Validate one-step image transitions; this is not a learner TODO."""

    expected_image_shape = (
        encoder.in_channels,
        encoder.image_size,
        encoder.image_size,
    )
    if current_images.ndim != 4 or current_images.shape[1:] != expected_image_shape:
        raise ValueError("current_images do not match the encoder image shape")
    if next_images.shape != current_images.shape:
        raise ValueError("next_images must have the same shape as current_images")
    if actions.shape != (current_images.shape[0], action_dim):
        raise ValueError("actions must have shape [batch, action_dim]")
    if (
        actions.device != current_images.device
        or next_images.device != current_images.device
    ):
        raise ValueError("transition tensors must be on the same device")


def validate_prediction_inputs(
    current_latents: torch.Tensor,
    actions: torch.Tensor,
    num_patches: int,
    embed_dim: int,
    action_dim: int,
) -> None:
    """Validate action-conditioned predictor inputs."""

    if current_latents.ndim != 3 or current_latents.shape[1:] != (
        num_patches,
        embed_dim,
    ):
        raise ValueError(
            "current_latents must have shape [batch, num_patches, embed_dim]"
        )
    if actions.shape != (current_latents.shape[0], action_dim):
        raise ValueError("actions must have shape [batch, action_dim]")
    if actions.device != current_latents.device:
        raise ValueError("actions and latents must be on the same device")


def validate_rollout_inputs(
    initial_latents: torch.Tensor,
    action_sequences: torch.Tensor,
    num_patches: int,
    embed_dim: int,
    action_dim: int,
) -> None:
    """Validate batched latent rollouts."""

    if initial_latents.ndim != 3 or initial_latents.shape[1:] != (
        num_patches,
        embed_dim,
    ):
        raise ValueError(
            "initial_latents must have shape [batch, num_patches, embed_dim]"
        )
    if (
        action_sequences.ndim != 3
        or action_sequences.shape[0] != initial_latents.shape[0]
    ):
        raise ValueError(
            "action_sequences must have shape [batch, horizon, action_dim]"
        )
    if action_sequences.shape[1] == 0 or action_sequences.shape[2] != action_dim:
        raise ValueError(
            "action_sequences must have positive horizon and matching action_dim"
        )
    if action_sequences.device != initial_latents.device:
        raise ValueError("actions and latents must be on the same device")


def prepare_goal_latents(
    predicted_latents: torch.Tensor,
    goal_latents: torch.Tensor,
) -> torch.Tensor:
    """Validate and broadcast goal latents to a candidate batch."""

    if predicted_latents.ndim != 3:
        raise ValueError(
            "predicted_latents must have shape [batch, patches, embed_dim]"
        )
    if goal_latents.ndim == 2:
        goal_latents = goal_latents.unsqueeze(0)
    if goal_latents.ndim != 3 or goal_latents.shape[1:] != predicted_latents.shape[1:]:
        raise ValueError("goal_latents must match the patch and embedding dimensions")
    if goal_latents.shape[0] == 1:
        goal_latents = goal_latents.expand(predicted_latents.shape[0], -1, -1)
    if goal_latents.shape != predicted_latents.shape:
        raise ValueError("goal_latents batch must be one or match predicted_latents")
    if goal_latents.device != predicted_latents.device:
        raise ValueError("goal and predicted latents must be on the same device")
    return goal_latents


def validate_planning_inputs(
    current_image: torch.Tensor,
    goal_image: torch.Tensor,
    candidate_actions: torch.Tensor,
    action_dim: int,
) -> None:
    """Validate short candidate planning inputs."""

    if current_image.ndim != 3 or goal_image.shape != current_image.shape:
        raise ValueError(
            "current_image and goal_image must share [channels, height, width]"
        )
    if candidate_actions.ndim != 3 or candidate_actions.shape[0] == 0:
        raise ValueError(
            "candidate_actions must have shape [candidates, horizon, action_dim]"
        )
    if candidate_actions.shape[1] == 0 or candidate_actions.shape[2] != action_dim:
        raise ValueError(
            "candidate actions need positive horizon and matching action_dim"
        )
    if (
        current_image.device != goal_image.device
        or candidate_actions.device != current_image.device
    ):
        raise ValueError("planning tensors must be on the same device")
