"""Reference solution for receding-horizon planning with a latent WAM."""

import torch

from provided import (
    CEMConfig,
    ControlTrace,
    PlanningResult,
    ProvidedJointWAM,
    TinyContinuousGrid,
    search_action_chunks,
    validate_control_inputs,
    validate_plan_inputs,
    validate_rollout_inputs,
    validate_score_inputs,
)


@torch.no_grad()
def rollout_latent_candidates(
    model: ProvidedJointWAM,
    initial_latent: torch.Tensor,
    action_chunks: torch.Tensor,
) -> torch.Tensor:
    """Imagine every candidate as [candidates, horizon + 1, state_dim]."""

    validate_rollout_inputs(model, initial_latent, action_chunks)
    current_latents = initial_latent.unsqueeze(0).expand(action_chunks.shape[0], -1)
    trajectory = [current_latents]
    for step in range(action_chunks.shape[1]):
        current_latents = model.predict_next_latent(
            current_latents,
            action_chunks[:, step],
        )
        trajectory.append(current_latents)
    return torch.stack(trajectory, dim=1)


def score_goal_progress(
    trajectories: torch.Tensor,
    goal_latent: torch.Tensor,
    action_chunks: torch.Tensor,
    action_penalty: float = 0.01,
) -> torch.Tensor:
    """Reward reduced goal distance and lightly penalize action effort."""

    validate_score_inputs(
        trajectories,
        goal_latent,
        action_chunks,
        action_penalty,
    )
    initial_distance = ((trajectories[:, 0] - goal_latent) ** 2).sum(dim=1)
    future_distances = ((trajectories[:, 1:] - goal_latent) ** 2).sum(dim=2)
    cumulative_progress = (initial_distance.unsqueeze(1) - future_distances).sum(dim=1)
    action_effort = (action_chunks**2).sum(dim=(1, 2))
    return cumulative_progress - action_penalty * action_effort


@torch.no_grad()
def select_action_chunk(
    model: ProvidedJointWAM,
    current_video: torch.Tensor,
    goal_instruction_ids: torch.Tensor,
    goal_instruction_mask: torch.Tensor,
    config: CEMConfig,
    action_penalty: float = 0.01,
    generator: torch.Generator | None = None,
) -> PlanningResult:
    """Search action chunks and return the best imagined trajectory."""

    validate_plan_inputs(
        model,
        current_video,
        goal_instruction_ids,
        goal_instruction_mask,
        config,
    )
    initial_latent = model.encode_video(current_video.unsqueeze(0))[0]
    goal_latent = model.encode_goal(
        goal_instruction_ids.unsqueeze(0),
        goal_instruction_mask.unsqueeze(0),
    )[0].to(device=initial_latent.device, dtype=initial_latent.dtype)

    def evaluate(action_chunks: torch.Tensor) -> torch.Tensor:
        trajectories = rollout_latent_candidates(
            model,
            initial_latent,
            action_chunks,
        )
        return score_goal_progress(
            trajectories,
            goal_latent,
            action_chunks,
            action_penalty,
        )

    search_result = search_action_chunks(
        evaluate,
        model.action_dim,
        config,
        generator=generator,
        device=initial_latent.device,
        dtype=initial_latent.dtype,
    )
    imagined_latents = rollout_latent_candidates(
        model,
        initial_latent,
        search_result.action_chunk.unsqueeze(0),
    )[0]
    return PlanningResult(
        action_chunk=search_result.action_chunk,
        imagined_latents=imagined_latents,
        score=search_result.score,
    )


@torch.no_grad()
def receding_horizon_control(
    model: ProvidedJointWAM,
    environment: TinyContinuousGrid,
    goal_instruction_ids: torch.Tensor,
    goal_instruction_mask: torch.Tensor,
    config: CEMConfig,
    max_steps: int,
    goal_tolerance: float = 0.15,
    action_penalty: float = 0.01,
    generator: torch.Generator | None = None,
) -> ControlTrace:
    """Execute only each planned chunk's first action, then observe and replan."""

    validate_control_inputs(
        model,
        environment,
        goal_instruction_ids,
        goal_instruction_mask,
        config,
        max_steps,
        goal_tolerance,
    )
    goal_latent = model.encode_goal(
        goal_instruction_ids.unsqueeze(0),
        goal_instruction_mask.unsqueeze(0),
    )[0].to(device=environment.position.device, dtype=environment.position.dtype)
    positions = [environment.position.clone()]
    actions = []
    planned_scores = []

    for _ in range(max_steps):
        if (
            torch.linalg.vector_norm(environment.position - goal_latent)
            <= goal_tolerance
        ):
            break
        plan = select_action_chunk(
            model,
            environment.observe(),
            goal_instruction_ids,
            goal_instruction_mask,
            config,
            action_penalty,
            generator,
        )
        first_action = plan.action_chunk[0]
        environment.step(first_action)
        actions.append(first_action.clone())
        planned_scores.append(plan.score.clone())
        positions.append(environment.position.clone())

    action_tensor = (
        torch.stack(actions)
        if actions
        else environment.position.new_empty((0, model.action_dim))
    )
    score_tensor = (
        torch.stack(planned_scores)
        if planned_scores
        else environment.position.new_empty((0,))
    )
    reached_goal = bool(
        torch.linalg.vector_norm(environment.position - goal_latent) <= goal_tolerance
    )
    return ControlTrace(
        positions=torch.stack(positions),
        actions=action_tensor,
        planned_scores=score_tensor,
        reached_goal=reached_goal,
    )
