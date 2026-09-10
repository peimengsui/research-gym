"""Provided branching world, stochastic WAM proxy, and validation for wm.13."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointMixturePrediction:
    """Mixture parameters over one flattened action-and-future strategy."""

    logits: torch.Tensor
    means: torch.Tensor
    log_scales: torch.Tensor


@dataclass(frozen=True)
class StrategyHypotheses:
    """Coherent strategies sampled using one component id per whole chunk."""

    component_ids: torch.Tensor
    action_chunks: torch.Tensor
    future_latents: torch.Tensor


@dataclass(frozen=True)
class SelectedStrategy:
    """Highest-scoring sampled strategy for each batch item."""

    component_ids: torch.Tensor
    action_chunks: torch.Tensor
    future_latents: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class BranchingProblem:
    """A tiny start, goal, and obstacle with two valid routes."""

    current_videos: torch.Tensor
    goal_instruction_ids: torch.Tensor
    goal_instruction_mask: torch.Tensor
    obstacle_latent: torch.Tensor


def render_position_videos(
    positions: torch.Tensor,
    grid_size: int = 3,
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
            images.scatter_add_(
                1,
                indices.unsqueeze(1),
                (row_weights * col_weights).unsqueeze(1),
            )
    images = images.reshape(positions.shape[0], 1, grid_size, grid_size)
    return images.unsqueeze(1).expand(-1, frame_count, -1, -1, -1).clone()


def make_goal_instruction(
    goal_positions: torch.Tensor, grid_size: int = 3
) -> torch.Tensor:
    """Encode integer goals as batched `[reach, row, column]` token ids."""

    if goal_positions.ndim != 2 or goal_positions.shape[1] != 2:
        raise ValueError("goal_positions must have shape [batch, 2]")
    if goal_positions.dtype != torch.long:
        raise ValueError("goal_positions must use torch.long")
    if (goal_positions < 0).any() or (goal_positions >= grid_size).any():
        raise ValueError("goal_positions must lie inside the grid")
    reach = goal_positions.new_ones((goal_positions.shape[0], 1))
    return torch.cat((reach, goal_positions + 2), dim=1)


class ProvidedStochasticWAM:
    """Exact frozen proxy for a WAM with a learned joint mixture head.

    The two mixture components encode routes around opposite sides of a central
    obstacle. Each component jointly describes an action chunk and its matching
    future latent trajectory.
    """

    state_dim = 2
    action_dim = 2
    horizon = 3
    num_mixtures = 2

    def __init__(
        self,
        grid_size: int = 3,
        frame_count: int = 2,
        component_log_scale: float = -3.0,
    ):
        if grid_size != 3:
            raise ValueError("the provided branching world uses grid_size=3")
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if not torch.isfinite(torch.tensor(component_log_scale)):
            raise ValueError("component_log_scale must be finite")
        self.grid_size = grid_size
        self.frame_count = frame_count
        self.component_log_scale = component_log_scale

    @property
    def strategy_dim(self) -> int:
        """Flattened action chunk plus future trajectory width."""

        return self.horizon * (self.action_dim + self.state_dim)

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        """Recover continuous dot positions as [batch, state_dim] latents."""

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
        coordinate_tokens = instruction_ids[:, 1:]
        if (coordinate_tokens < 2).any() or (
            coordinate_tokens >= self.grid_size + 2
        ).any():
            raise ValueError("goal coordinate token is outside the grid vocabulary")
        return (coordinate_tokens - 2).to(torch.float32)

    def predict_strategy_mixture(
        self,
        current_latents: torch.Tensor,
        goal_latents: torch.Tensor,
    ) -> JointMixturePrediction:
        """Return left-route and right-route joint Gaussian components."""

        expected_shape = (current_latents.shape[0], self.state_dim)
        if current_latents.ndim != 2 or current_latents.shape[1] != self.state_dim:
            raise ValueError("current_latents must have shape [batch, state_dim]")
        if goal_latents.shape != expected_shape:
            raise ValueError("goal_latents must match current_latents")
        if current_latents.device != goal_latents.device:
            raise ValueError("current and goal latents must share a device")
        if current_latents.dtype != goal_latents.dtype:
            raise ValueError("current and goal latents must share a dtype")
        if not torch.is_floating_point(current_latents):
            raise ValueError("current and goal latents must be floating point")
        if (
            not torch.isfinite(current_latents).all()
            or not torch.isfinite(goal_latents).all()
        ):
            raise ValueError("current and goal latents must be finite")

        expected_start = current_latents.new_tensor([0.0, 1.0]).expand_as(
            current_latents
        )
        expected_goal = goal_latents.new_tensor([2.0, 1.0]).expand_as(goal_latents)
        if not torch.allclose(current_latents, expected_start, atol=1e-5):
            raise ValueError("the branching example expects start latent [0, 1]")
        if not torch.allclose(goal_latents, expected_goal, atol=1e-5):
            raise ValueError("the branching example expects goal latent [2, 1]")

        batch_size = current_latents.shape[0]
        waypoint_rows = current_latents.new_ones((batch_size, self.num_mixtures))
        waypoint_cols = current_latents.new_tensor([0.0, 2.0]).expand(batch_size, -1)
        waypoints = torch.stack((waypoint_rows, waypoint_cols), dim=2)
        repeated_goals = goal_latents.unsqueeze(1).expand(-1, self.num_mixtures, -1)
        future_latents = torch.stack(
            (waypoints, repeated_goals, repeated_goals),
            dim=2,
        )
        initial = current_latents[:, None, None, :].expand(
            -1,
            self.num_mixtures,
            1,
            -1,
        )
        previous_latents = torch.cat((initial, future_latents[:, :, :-1]), dim=2)
        action_chunks = future_latents - previous_latents
        means = torch.cat(
            (
                action_chunks.flatten(start_dim=2),
                future_latents.flatten(start_dim=2),
            ),
            dim=2,
        )
        logits = current_latents.new_zeros((batch_size, self.num_mixtures))
        log_scales = torch.full_like(means, self.component_log_scale)
        return JointMixturePrediction(logits, means, log_scales)

    def __call__(
        self,
        current_videos: torch.Tensor,
        goal_instruction_ids: torch.Tensor,
        goal_instruction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, JointMixturePrediction]:
        """Encode observations and return their joint strategy mixtures."""

        current_latents = self.encode_video(current_videos)
        goal_latents = self.encode_goal(
            goal_instruction_ids,
            goal_instruction_mask,
        ).to(device=current_latents.device, dtype=current_latents.dtype)
        prediction = self.predict_strategy_mixture(current_latents, goal_latents)
        return current_latents, goal_latents, prediction


def make_branching_problem() -> BranchingProblem:
    """Return a start and goal separated by a central obstacle."""

    current_position = torch.tensor([[0.0, 1.0]])
    goal_position = torch.tensor([[2, 1]])
    goal_ids = make_goal_instruction(goal_position)
    return BranchingProblem(
        current_videos=render_position_videos(current_position),
        goal_instruction_ids=goal_ids,
        goal_instruction_mask=torch.ones_like(goal_ids, dtype=torch.bool),
        obstacle_latent=torch.tensor([1.0, 1.0]),
    )


def unpack_strategy_vectors(
    vectors: torch.Tensor,
    horizon: int,
    action_dim: int,
    state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split [..., strategy_dim] vectors into action and future tensors."""

    if horizon <= 0 or action_dim <= 0 or state_dim <= 0:
        raise ValueError("horizon, action_dim, and state_dim must be positive")
    expected_dim = horizon * (action_dim + state_dim)
    if vectors.ndim < 1 or vectors.shape[-1] != expected_dim:
        raise ValueError("vectors do not match the requested strategy dimensions")
    action_width = horizon * action_dim
    action_chunks = vectors[..., :action_width].reshape(
        *vectors.shape[:-1],
        horizon,
        action_dim,
    )
    future_latents = vectors[..., action_width:].reshape(
        *vectors.shape[:-1],
        horizon,
        state_dim,
    )
    return action_chunks, future_latents


def mixture_mean_strategy(prediction: JointMixturePrediction) -> torch.Tensor:
    """Return the probability-weighted mean, for comparison only."""

    if prediction.logits.ndim != 2 or prediction.means.ndim != 3:
        raise ValueError("prediction must contain batched mixture tensors")
    probabilities = torch.softmax(prediction.logits, dim=1).unsqueeze(2)
    return (probabilities * prediction.means).sum(dim=1)


def strategy_hits_obstacle(
    future_latents: torch.Tensor,
    obstacle_latent: torch.Tensor,
    collision_radius: float,
) -> torch.Tensor:
    """Return [...]-shaped flags for trajectories entering obstacle radius."""

    if future_latents.ndim < 3:
        raise ValueError("future_latents must end with [horizon, state_dim]")
    if obstacle_latent.shape != (future_latents.shape[-1],):
        raise ValueError("obstacle_latent must have shape [state_dim]")
    if obstacle_latent.device != future_latents.device:
        raise ValueError("future latents and obstacle must share a device")
    if collision_radius < 0.0:
        raise ValueError("collision_radius must be non-negative")
    squared_distances = ((future_latents - obstacle_latent) ** 2).sum(dim=-1)
    return (squared_distances <= collision_radius**2).any(dim=-1)


def validate_pack_inputs(
    action_chunks: torch.Tensor,
    future_latents: torch.Tensor,
) -> None:
    """Validate aligned action and future targets."""

    if action_chunks.ndim != 3 or future_latents.ndim != 3:
        raise ValueError("strategy targets must be rank-three tensors")
    if action_chunks.shape[:2] != future_latents.shape[:2]:
        raise ValueError(
            "action chunks and future latents must align in batch and time"
        )
    if action_chunks.shape[0] == 0 or action_chunks.shape[1] == 0:
        raise ValueError("strategy targets need positive batch and horizon")
    if action_chunks.shape[2] == 0 or future_latents.shape[2] == 0:
        raise ValueError("action_dim and state_dim must be positive")
    if action_chunks.device != future_latents.device:
        raise ValueError("strategy targets must share a device")
    if action_chunks.dtype != future_latents.dtype:
        raise ValueError("strategy targets must share a dtype")
    if not torch.is_floating_point(action_chunks):
        raise ValueError("strategy targets must be floating point")
    if (
        not torch.isfinite(action_chunks).all()
        or not torch.isfinite(future_latents).all()
    ):
        raise ValueError("strategy targets must be finite")


def validate_mixture_prediction(
    prediction: JointMixturePrediction,
    strategy_dim: int | None = None,
) -> None:
    """Validate joint Gaussian mixture parameter tensors."""

    if prediction.logits.ndim != 2:
        raise ValueError("mixture logits must have shape [batch, mixtures]")
    if prediction.means.ndim != 3:
        raise ValueError(
            "mixture means must have shape [batch, mixtures, strategy_dim]"
        )
    if prediction.log_scales.shape != prediction.means.shape:
        raise ValueError("log_scales must match mixture means")
    if prediction.means.shape[:2] != prediction.logits.shape:
        raise ValueError(
            "mixture parameters must align in batch and component dimensions"
        )
    if prediction.logits.shape[0] == 0 or prediction.logits.shape[1] == 0:
        raise ValueError("mixture prediction needs positive batch and component counts")
    if prediction.means.shape[2] == 0:
        raise ValueError("strategy_dim must be positive")
    if strategy_dim is not None and prediction.means.shape[2] != strategy_dim:
        raise ValueError("mixture strategy dimension does not match targets")
    tensors = (prediction.logits, prediction.means, prediction.log_scales)
    if any(tensor.device != prediction.logits.device for tensor in tensors):
        raise ValueError("mixture parameters must share a device")
    if any(tensor.dtype != prediction.logits.dtype for tensor in tensors):
        raise ValueError("mixture parameters must share a dtype")
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("mixture parameters must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("mixture parameters must be finite")


def validate_loss_inputs(
    prediction: JointMixturePrediction,
    action_chunks: torch.Tensor,
    future_latents: torch.Tensor,
) -> None:
    """Validate mixture and target tensors before learner loss code runs."""

    validate_pack_inputs(action_chunks, future_latents)
    strategy_dim = action_chunks.shape[1] * (
        action_chunks.shape[2] + future_latents.shape[2]
    )
    validate_mixture_prediction(prediction, strategy_dim)
    if prediction.logits.shape[0] != action_chunks.shape[0]:
        raise ValueError("prediction and target batch sizes must match")
    if prediction.logits.device != action_chunks.device:
        raise ValueError("prediction and targets must share a device")
    if prediction.logits.dtype != action_chunks.dtype:
        raise ValueError("prediction and targets must share a dtype")


def validate_sampling_inputs(
    prediction: JointMixturePrediction,
    num_samples: int,
    horizon: int,
    action_dim: int,
    state_dim: int,
) -> None:
    """Validate mixture dimensions for coherent sampling."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if horizon <= 0 or action_dim <= 0 or state_dim <= 0:
        raise ValueError("strategy dimensions must be positive")
    strategy_dim = horizon * (action_dim + state_dim)
    validate_mixture_prediction(prediction, strategy_dim)


def validate_hypotheses(hypotheses: StrategyHypotheses) -> None:
    """Validate sampled component, action, and future shapes."""

    if hypotheses.component_ids.ndim != 2:
        raise ValueError("component_ids must have shape [batch, samples]")
    if hypotheses.component_ids.dtype != torch.long:
        raise ValueError("component_ids must use torch.long")
    if hypotheses.action_chunks.ndim != 4 or hypotheses.future_latents.ndim != 4:
        raise ValueError("sampled actions and futures must be rank-four tensors")
    expected_prefix = hypotheses.component_ids.shape
    if hypotheses.action_chunks.shape[:2] != expected_prefix:
        raise ValueError("action hypotheses must align with component ids")
    if hypotheses.future_latents.shape[:2] != expected_prefix:
        raise ValueError("future hypotheses must align with component ids")
    if hypotheses.action_chunks.shape[2] != hypotheses.future_latents.shape[2]:
        raise ValueError("sampled actions and futures must share a horizon")
    tensors = (
        hypotheses.component_ids,
        hypotheses.action_chunks,
        hypotheses.future_latents,
    )
    if any(tensor.device != hypotheses.component_ids.device for tensor in tensors):
        raise ValueError("sampled hypothesis tensors must share a device")
    if hypotheses.action_chunks.dtype != hypotheses.future_latents.dtype:
        raise ValueError("sampled actions and futures must share a dtype")
    if not torch.is_floating_point(hypotheses.action_chunks):
        raise ValueError("sampled actions and futures must be floating point")
    if (
        not torch.isfinite(hypotheses.action_chunks).all()
        or not torch.isfinite(hypotheses.future_latents).all()
    ):
        raise ValueError("sampled hypotheses must be finite")


def validate_scoring_inputs(
    hypotheses: StrategyHypotheses,
    current_latents: torch.Tensor,
    goal_latents: torch.Tensor,
    obstacle_latent: torch.Tensor,
    collision_radius: float,
    collision_penalty: float,
    coherence_penalty: float,
    action_penalty: float,
) -> None:
    """Validate tensors and non-negative strategy costs."""

    validate_hypotheses(hypotheses)
    batch_size = hypotheses.component_ids.shape[0]
    state_dim = hypotheses.future_latents.shape[3]
    if current_latents.shape != (batch_size, state_dim):
        raise ValueError("current_latents must have shape [batch, state_dim]")
    if goal_latents.shape != current_latents.shape:
        raise ValueError("goal_latents must match current_latents")
    if obstacle_latent.shape != (state_dim,):
        raise ValueError("obstacle_latent must have shape [state_dim]")
    if hypotheses.action_chunks.shape[3] != state_dim:
        raise ValueError("this tiny world expects action_dim to equal state_dim")
    tensors = (current_latents, goal_latents, obstacle_latent)
    if any(tensor.device != hypotheses.component_ids.device for tensor in tensors):
        raise ValueError("scoring tensors must share the hypotheses device")
    if any(tensor.dtype != hypotheses.action_chunks.dtype for tensor in tensors):
        raise ValueError("floating-point scoring tensors must share a dtype")
    penalties = (
        collision_radius,
        collision_penalty,
        coherence_penalty,
        action_penalty,
    )
    if any(value < 0.0 for value in penalties):
        raise ValueError("strategy radii and penalties must be non-negative")


def validate_selection_inputs(
    hypotheses: StrategyHypotheses,
    scores: torch.Tensor,
) -> None:
    """Validate one score per sampled hypothesis."""

    validate_hypotheses(hypotheses)
    if scores.shape != hypotheses.component_ids.shape:
        raise ValueError("scores must have shape [batch, samples]")
    if scores.device != hypotheses.component_ids.device:
        raise ValueError("scores and hypotheses must share a device")
    if scores.dtype != hypotheses.action_chunks.dtype:
        raise ValueError("scores must match hypothesis floating-point dtype")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
