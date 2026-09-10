"""Provided WAM, continuous grid, CEM search, and validation for wm.12."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CEMConfig:
    """Configuration for the provided action-chunk search."""

    horizon: int
    num_iterations: int = 5
    num_samples: int = 256
    num_elites: int = 32
    initial_std: float = 1.0
    min_std: float = 0.05
    action_low: float = -1.0
    action_high: float = 1.0


@dataclass(frozen=True)
class CEMSearchResult:
    """Best action chunk and score found across CEM iterations."""

    action_chunk: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True)
class PlanningResult:
    """Selected action chunk and its imagined latent trajectory."""

    action_chunk: torch.Tensor
    imagined_latents: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True)
class ControlTrace:
    """Observed positions and first actions executed during replanning."""

    positions: torch.Tensor
    actions: torch.Tensor
    planned_scores: torch.Tensor
    reached_goal: bool


def render_position_videos(
    positions: torch.Tensor,
    grid_size: int = 4,
    frame_count: int = 2,
) -> torch.Tensor:
    """Render continuous [row, column] positions with bilinear dot weights."""

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [batch, 2]")
    if not torch.is_floating_point(positions) or not torch.isfinite(positions).all():
        raise ValueError("positions must be finite floating-point values")
    if grid_size <= 1 or frame_count <= 0:
        raise ValueError("grid_size must exceed one and frame_count must be positive")
    if (positions < 0.0).any() or (positions > grid_size - 1).any():
        raise ValueError("positions must lie inside the grid")

    lower = positions.floor().to(torch.long)
    upper = (lower + 1).clamp_max(grid_size - 1)
    fraction = positions - lower.to(positions.dtype)
    row_choices = ((lower[:, 0], 1.0 - fraction[:, 0]), (upper[:, 0], fraction[:, 0]))
    col_choices = ((lower[:, 1], 1.0 - fraction[:, 1]), (upper[:, 1], fraction[:, 1]))
    images = torch.zeros(
        positions.shape[0],
        grid_size * grid_size,
        device=positions.device,
        dtype=positions.dtype,
    )
    for rows, row_weights in row_choices:
        for cols, col_weights in col_choices:
            indices = rows * grid_size + cols
            weights = row_weights * col_weights
            images.scatter_add_(1, indices.unsqueeze(1), weights.unsqueeze(1))
    images = images.reshape(positions.shape[0], 1, grid_size, grid_size)
    return images.unsqueeze(1).expand(-1, frame_count, -1, -1, -1).clone()


class ProvidedJointWAM:
    """Exact frozen stand-in for the converged latent interface from wm.11.

    The model keeps the same two planning-facing methods: encode a video into a
    state latent and predict a next latent from a state/action pair. Exact
    dynamics make this lesson about planning rather than model fitting error.
    """

    state_dim = 2
    action_dim = 2

    def __init__(self, grid_size: int = 4, frame_count: int = 2):
        if grid_size <= 1 or frame_count <= 0:
            raise ValueError(
                "grid_size must exceed one and frame_count must be positive"
            )
        self.grid_size = grid_size
        self.frame_count = frame_count

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        """Recover continuous dot positions as [batch, 2] state latents."""

        expected = (self.frame_count, 1, self.grid_size, self.grid_size)
        if videos.ndim != 5 or videos.shape[1:] != expected:
            raise ValueError(
                "videos must have shape [batch, frame_count, 1, grid_size, grid_size]"
            )
        if not torch.is_floating_point(videos) or not torch.isfinite(videos).all():
            raise ValueError("videos must be finite floating-point values")
        frame = videos[:, -1, 0]
        if (frame < 0.0).any():
            raise ValueError("rendered dot weights must be non-negative")
        mass = frame.sum(dim=(1, 2))
        if not torch.allclose(mass, torch.ones_like(mass), atol=1e-5):
            raise ValueError("each video must contain one unit of dot mass")
        coordinates = torch.arange(
            self.grid_size,
            device=videos.device,
            dtype=videos.dtype,
        )
        rows = (frame.sum(dim=2) * coordinates).sum(dim=1)
        cols = (frame.sum(dim=1) * coordinates).sum(dim=1)
        return torch.stack((rows, cols), dim=1)

    def encode_goal(
        self,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode `reach <row> <column>` tokens into matching state latents."""

        if instruction_ids.ndim != 2 or instruction_ids.shape[1] != 3:
            raise ValueError("goal instructions must have shape [batch, 3]")
        if instruction_ids.dtype != torch.long:
            raise ValueError("goal instruction ids must use torch.long")
        if instruction_mask.shape != instruction_ids.shape:
            raise ValueError("instruction_mask must match instruction_ids")
        if instruction_mask.dtype != torch.bool or not instruction_mask.all():
            raise ValueError("all three goal instruction tokens must be valid")
        if instruction_ids.device != instruction_mask.device:
            raise ValueError("goal instruction tensors must share a device")
        if not (instruction_ids[:, 0] == 1).all():
            raise ValueError("goal instructions must begin with the reach token")
        value_tokens = instruction_ids[:, 1:]
        if (value_tokens < 2).any() or (value_tokens >= self.grid_size + 2).any():
            raise ValueError("goal coordinate token is outside the grid vocabulary")
        return (value_tokens - 2).to(torch.float32)

    def predict_next_latent(
        self,
        state_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply bounded continuous actions in the exact latent grid."""

        if state_latents.ndim != 2 or state_latents.shape[1] != self.state_dim:
            raise ValueError("state_latents must have shape [batch, state_dim]")
        if actions.shape != (state_latents.shape[0], self.action_dim):
            raise ValueError("actions must have shape [batch, action_dim]")
        if actions.device != state_latents.device:
            raise ValueError("states and actions must share a device")
        if not torch.is_floating_point(actions) or not torch.isfinite(actions).all():
            raise ValueError("actions must be finite floating-point values")
        if (actions < -1.0).any() or (actions > 1.0).any():
            raise ValueError("actions must lie in [-1, 1]")
        return (state_latents + actions).clamp(0.0, float(self.grid_size - 1))


class TinyContinuousGrid:
    """Small observation loop used to demonstrate receding-horizon control."""

    def __init__(
        self,
        initial_position: torch.Tensor,
        grid_size: int = 4,
        frame_count: int = 2,
    ):
        if initial_position.shape != (2,):
            raise ValueError("initial_position must have shape [2]")
        if not torch.is_floating_point(initial_position):
            raise ValueError("initial_position must be floating point")
        if not torch.isfinite(initial_position).all():
            raise ValueError("initial_position must be finite")
        if grid_size <= 1 or frame_count <= 0:
            raise ValueError(
                "grid_size must exceed one and frame_count must be positive"
            )
        if (initial_position < 0.0).any() or (initial_position > grid_size - 1).any():
            raise ValueError("initial_position must lie inside the grid")
        self.grid_size = grid_size
        self.frame_count = frame_count
        self.position = initial_position.clone()

    def observe(self) -> torch.Tensor:
        """Return an unbatched [frames, 1, grid, grid] video observation."""

        return render_position_videos(
            self.position.unsqueeze(0),
            self.grid_size,
            self.frame_count,
        )[0]

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """Execute one bounded [delta_row, delta_column] action."""

        if action.shape != (2,) or not torch.is_floating_point(action):
            raise ValueError("action must be a floating-point [2] tensor")
        if (
            not torch.isfinite(action).all()
            or (action < -1.0).any()
            or (action > 1.0).any()
        ):
            raise ValueError("action must be finite and lie in [-1, 1]")
        if action.device != self.position.device:
            raise ValueError("action and environment position must share a device")
        self.position = (self.position + action).clamp(
            0.0,
            float(self.grid_size - 1),
        )
        return self.observe()


def make_goal_instruction(
    goal_position: torch.Tensor, grid_size: int = 4
) -> torch.Tensor:
    """Encode an integer grid goal as `[reach, row, column]` token ids."""

    if goal_position.shape != (2,) or goal_position.dtype != torch.long:
        raise ValueError("goal_position must be a long [2] tensor")
    if (goal_position < 0).any() or (goal_position >= grid_size).any():
        raise ValueError("goal_position must lie inside the grid")
    return torch.cat((goal_position.new_ones(1), goal_position + 2))


def validate_cem_config(config: CEMConfig) -> None:
    """Validate provided CEM settings."""

    if config.horizon <= 0 or config.num_iterations <= 0 or config.num_samples <= 0:
        raise ValueError("CEM horizon, iterations, and samples must be positive")
    if config.num_elites <= 0 or config.num_elites > config.num_samples:
        raise ValueError("num_elites must be in [1, num_samples]")
    if config.initial_std <= 0.0 or config.min_std <= 0.0:
        raise ValueError("CEM standard deviations must be positive")
    if config.action_high <= config.action_low:
        raise ValueError("action_high must exceed action_low")


@torch.no_grad()
def search_action_chunks(
    evaluate: Callable[[torch.Tensor], torch.Tensor],
    action_dim: int,
    config: CEMConfig,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> CEMSearchResult:
    """Run provided CEM updates around a learner-supplied scoring function."""

    validate_cem_config(config)
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    mean = torch.zeros(config.horizon, action_dim, device=device, dtype=dtype)
    std = torch.full_like(mean, config.initial_std)
    best_score = torch.tensor(float("-inf"), device=device, dtype=dtype)
    best_actions = mean.clone()

    for _ in range(config.num_iterations):
        noise = torch.randn(
            config.num_samples,
            config.horizon,
            action_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        candidates = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
            config.action_low,
            config.action_high,
        )
        scores = evaluate(candidates)
        if scores.shape != (config.num_samples,) or not torch.isfinite(scores).all():
            raise ValueError("evaluate must return one finite score per CEM sample")
        iteration_score, iteration_index = scores.max(dim=0)
        if iteration_score > best_score:
            best_score = iteration_score.clone()
            best_actions = candidates[iteration_index].clone()
        elite_indices = scores.topk(config.num_elites).indices
        elites = candidates[elite_indices]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0, unbiased=False).clamp_min(config.min_std)
    return CEMSearchResult(best_actions, best_score)


def validate_rollout_inputs(
    model: ProvidedJointWAM,
    initial_latent: torch.Tensor,
    action_chunks: torch.Tensor,
) -> None:
    """Validate candidate rollout tensors before learner code runs."""

    if not isinstance(model, ProvidedJointWAM):
        raise TypeError("model must be a ProvidedJointWAM")
    if initial_latent.shape != (model.state_dim,):
        raise ValueError("initial_latent must have shape [state_dim]")
    if (
        not torch.is_floating_point(initial_latent)
        or not torch.isfinite(initial_latent).all()
    ):
        raise ValueError("initial_latent must contain finite floating-point values")
    if action_chunks.ndim != 3 or action_chunks.shape[0] == 0:
        raise ValueError(
            "action_chunks must have shape [candidates, horizon, action_dim]"
        )
    if action_chunks.shape[1] == 0 or action_chunks.shape[2] != model.action_dim:
        raise ValueError("action chunks need positive horizon and matching action_dim")
    if action_chunks.device != initial_latent.device:
        raise ValueError("initial latent and action chunks must share a device")
    if action_chunks.dtype != initial_latent.dtype:
        raise ValueError("initial latent and action chunks must share a dtype")
    if (
        not torch.is_floating_point(action_chunks)
        or not torch.isfinite(action_chunks).all()
    ):
        raise ValueError("action_chunks must contain finite floating-point values")
    if (action_chunks < -1.0).any() or (action_chunks > 1.0).any():
        raise ValueError("action_chunks must lie in [-1, 1]")


def validate_score_inputs(
    trajectories: torch.Tensor,
    goal_latent: torch.Tensor,
    action_chunks: torch.Tensor,
    action_penalty: float,
) -> None:
    """Validate goal-progress scoring tensors."""

    if trajectories.ndim != 3 or trajectories.shape[0] == 0:
        raise ValueError(
            "trajectories must have shape [candidates, horizon + 1, state_dim]"
        )
    if action_chunks.ndim != 3 or action_chunks.shape[0] != trajectories.shape[0]:
        raise ValueError("action_chunks must align with candidate trajectories")
    if trajectories.shape[1] != action_chunks.shape[1] + 1:
        raise ValueError("trajectories must contain one more step than action chunks")
    if goal_latent.shape != (trajectories.shape[2],):
        raise ValueError("goal_latent must have shape [state_dim]")
    if action_chunks.shape[2] != trajectories.shape[2]:
        raise ValueError("this tiny grid expects action_dim to equal state_dim")
    if (
        goal_latent.device != trajectories.device
        or action_chunks.device != trajectories.device
    ):
        raise ValueError("score tensors must share a device")
    if (
        goal_latent.dtype != trajectories.dtype
        or action_chunks.dtype != trajectories.dtype
    ):
        raise ValueError("score tensors must share a dtype")
    tensors = (trajectories, goal_latent, action_chunks)
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("score tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("score tensors must be finite")
    if action_penalty < 0.0:
        raise ValueError("action_penalty must be non-negative")


def validate_plan_inputs(
    model: ProvidedJointWAM,
    current_video: torch.Tensor,
    goal_instruction_ids: torch.Tensor,
    goal_instruction_mask: torch.Tensor,
    config: CEMConfig,
) -> None:
    """Validate one planning query."""

    expected_video_shape = (model.frame_count, 1, model.grid_size, model.grid_size)
    if current_video.shape != expected_video_shape:
        raise ValueError("current_video does not match the WAM observation shape")
    if goal_instruction_ids.shape != (3,) or goal_instruction_ids.dtype != torch.long:
        raise ValueError("goal_instruction_ids must be a long [3] tensor")
    if goal_instruction_mask.shape != (3,) or goal_instruction_mask.dtype != torch.bool:
        raise ValueError("goal_instruction_mask must be a boolean [3] tensor")
    tensors = (goal_instruction_ids, goal_instruction_mask)
    if any(tensor.device != current_video.device for tensor in tensors):
        raise ValueError("planning inputs must share a device")
    validate_cem_config(config)


def validate_control_inputs(
    model: ProvidedJointWAM,
    environment: TinyContinuousGrid,
    goal_instruction_ids: torch.Tensor,
    goal_instruction_mask: torch.Tensor,
    config: CEMConfig,
    max_steps: int,
    goal_tolerance: float,
) -> None:
    """Validate a receding-horizon control request."""

    if environment.grid_size != model.grid_size:
        raise ValueError("environment and model grid sizes must match")
    if environment.frame_count != model.frame_count:
        raise ValueError("environment and model frame counts must match")
    validate_plan_inputs(
        model,
        environment.observe(),
        goal_instruction_ids,
        goal_instruction_mask,
        config,
    )
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if goal_tolerance < 0.0:
        raise ValueError("goal_tolerance must be non-negative")
