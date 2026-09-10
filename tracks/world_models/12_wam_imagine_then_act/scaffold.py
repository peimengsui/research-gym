"""Learner scaffold for receding-horizon planning with a latent WAM."""

import torch

from provided import (
    CEMConfig,
    ControlTrace,
    PlanningResult,
    ProvidedJointWAM,
    TinyContinuousGrid,
    search_action_chunks,  # noqa: F401 - used by TODO 3
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
    """Imagine every candidate as [candidates, horizon + 1, state_dim].

    initial_latent: [state_dim]
    action_chunks: [candidates, horizon, action_dim]
    returns: [candidates, horizon + 1, state_dim]
    """

    validate_rollout_inputs(model, initial_latent, action_chunks)
    # TODO 1: expand the initial latent across candidates. Repeatedly call
    # model.predict_next_latent with one action step, preserve the initial
    # latent at trajectory index zero, and stack along the time dimension.
    raise NotImplementedError("TODO: imagine all candidate action chunks")


def score_goal_progress(
    trajectories: torch.Tensor,
    goal_latent: torch.Tensor,
    action_chunks: torch.Tensor,
    action_penalty: float = 0.01,
) -> torch.Tensor:
    """Reward reduced goal distance and lightly penalize action effort.

    trajectories: [candidates, horizon + 1, state_dim]
    goal_latent: [state_dim]
    action_chunks: [candidates, horizon, action_dim]
    returns: [candidates], where higher is better
    """

    validate_score_inputs(
        trajectories,
        goal_latent,
        action_chunks,
        action_penalty,
    )
    # TODO 2: compute squared goal distance at the initial latent and at every
    # future latent. Sum (initial_distance - future_distance) across future
    # steps, then subtract action_penalty times each candidate's sum of squared
    # actions. Summing per-step progress rewards plans that improve earlier.
    raise NotImplementedError("TODO: score imagined goal progress")


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
    # TODO 3: encode the current video and language goal. Define an evaluate
    # closure that rolls out candidate chunks and scores goal progress. Pass it
    # to the provided search_action_chunks CEM helper, then imagine the selected
    # chunk once more and return PlanningResult. Keep tensors on the current
    # latent's device and dtype.
    raise NotImplementedError("TODO: select the best imagined action chunk")


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
    # TODO 4: encode the goal and record the initial position. Until the goal
    # tolerance or max_steps is reached: observe, call select_action_chunk,
    # execute only plan.action_chunk[0], and record that action, score, and new
    # position. Return rectangular tensors in ControlTrace; use empty tensors
    # with shapes [0, action_dim] and [0] if no action was needed.
    raise NotImplementedError("TODO: execute first actions and replan")
