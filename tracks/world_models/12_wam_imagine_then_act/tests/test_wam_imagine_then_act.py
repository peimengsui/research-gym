import pytest
import torch

import implementation
from implementation import (
    receding_horizon_control,
    rollout_latent_candidates,
    score_goal_progress,
    select_action_chunk,
)
from provided import (
    CEMConfig,
    PlanningResult,
    ProvidedJointWAM,
    TinyContinuousGrid,
    make_goal_instruction,
    render_position_videos,
)


def make_planning_inputs(
    initial_position: torch.Tensor | None = None,
    goal_position: torch.Tensor | None = None,
) -> tuple[ProvidedJointWAM, TinyContinuousGrid, torch.Tensor, torch.Tensor]:
    if initial_position is None:
        initial_position = torch.tensor([0.0, 0.0])
    if goal_position is None:
        goal_position = torch.tensor([3, 2])
    model = ProvidedJointWAM()
    environment = TinyContinuousGrid(initial_position)
    goal_ids = make_goal_instruction(goal_position)
    goal_mask = torch.ones_like(goal_ids, dtype=torch.bool)
    return model, environment, goal_ids, goal_mask


def test_provided_video_encoder_recovers_continuous_positions() -> None:
    model = ProvidedJointWAM()
    positions = torch.tensor([[0.25, 1.75], [3.0, 0.0]])

    videos = render_position_videos(positions)
    encoded = model.encode_video(videos)

    assert videos.shape == (2, 2, 1, 4, 4)
    assert torch.allclose(encoded, positions)


def test_goal_instruction_encodes_the_same_latent_coordinate() -> None:
    model, _, goal_ids, goal_mask = make_planning_inputs()

    goal_latent = model.encode_goal(goal_ids.unsqueeze(0), goal_mask.unsqueeze(0))

    assert torch.equal(goal_ids, torch.tensor([1, 5, 4]))
    assert torch.equal(goal_latent, torch.tensor([[3.0, 2.0]]))


def test_rollout_includes_initial_latent_and_clamps_grid_boundaries() -> None:
    model = ProvidedJointWAM()
    initial = torch.tensor([0.5, 0.5])
    action_chunks = torch.tensor(
        [
            [[1.0, 0.5], [1.0, 0.5]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )

    trajectories = rollout_latent_candidates(model, initial, action_chunks)

    expected = torch.tensor(
        [
            [[0.5, 0.5], [1.5, 1.0], [2.5, 1.5]],
            [[0.5, 0.5], [0.0, 0.5], [0.0, 0.0]],
        ]
    )
    assert trajectories.shape == (2, 3, 2)
    assert torch.allclose(trajectories, expected)
    assert not trajectories.requires_grad


def test_goal_progress_rewards_distance_reduction_minus_action_effort() -> None:
    trajectories = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    actions = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )

    scores = score_goal_progress(
        trajectories,
        torch.tensor([2.0, 0.0]),
        actions,
        action_penalty=0.1,
    )

    assert torch.allclose(scores, torch.tensor([6.8, 0.0]))


def test_select_action_chunk_imagines_progress_toward_goal() -> None:
    model, environment, goal_ids, goal_mask = make_planning_inputs()
    config = CEMConfig(
        horizon=3,
        num_iterations=5,
        num_samples=256,
        num_elites=32,
    )

    result = select_action_chunk(
        model,
        environment.observe(),
        goal_ids,
        goal_mask,
        config,
        generator=torch.Generator().manual_seed(7),
    )

    initial_distance = torch.linalg.vector_norm(torch.tensor([3.0, 2.0]))
    final_distance = torch.linalg.vector_norm(
        result.imagined_latents[-1] - torch.tensor([3.0, 2.0])
    )
    assert result.action_chunk.shape == (3, 2)
    assert result.imagined_latents.shape == (4, 2)
    assert final_distance < initial_distance * 0.1
    assert result.score > 0.0
    assert not result.action_chunk.requires_grad


def test_receding_horizon_control_reaches_goal() -> None:
    model, environment, goal_ids, goal_mask = make_planning_inputs()
    config = CEMConfig(
        horizon=3,
        num_iterations=5,
        num_samples=256,
        num_elites=32,
    )

    trace = receding_horizon_control(
        model,
        environment,
        goal_ids,
        goal_mask,
        config,
        max_steps=10,
        generator=torch.Generator().manual_seed(11),
    )

    assert trace.reached_goal
    assert trace.positions.shape[0] == trace.actions.shape[0] + 1
    assert trace.planned_scores.shape == (trace.actions.shape[0],)
    assert (
        torch.linalg.vector_norm(trace.positions[-1] - torch.tensor([3.0, 2.0])) < 0.15
    )
    assert not trace.actions.requires_grad


def test_controller_executes_only_first_action_then_replans(monkeypatch) -> None:
    model, environment, goal_ids, goal_mask = make_planning_inputs(
        goal_position=torch.tensor([1, 1])
    )
    config = CEMConfig(horizon=2)
    planned_chunks = [
        torch.tensor([[1.0, 0.0], [-99.0, -99.0]]),
        torch.tensor([[0.0, 1.0], [99.0, 99.0]]),
    ]
    planning_positions: list[torch.Tensor] = []

    def fake_select_action_chunk(*args, **kwargs) -> PlanningResult:
        del args, kwargs
        planning_positions.append(environment.position.clone())
        chunk = planned_chunks[len(planning_positions) - 1]
        return PlanningResult(
            action_chunk=chunk,
            imagined_latents=torch.zeros(3, 2),
            score=torch.tensor(1.0),
        )

    monkeypatch.setattr(implementation, "select_action_chunk", fake_select_action_chunk)

    trace = receding_horizon_control(
        model,
        environment,
        goal_ids,
        goal_mask,
        config,
        max_steps=2,
    )

    assert trace.reached_goal
    assert torch.equal(trace.actions, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(
        torch.stack(planning_positions),
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
    )


def test_controller_returns_rectangular_empty_history_at_goal() -> None:
    goal = torch.tensor([2, 1])
    model, environment, goal_ids, goal_mask = make_planning_inputs(
        initial_position=goal.to(torch.float32),
        goal_position=goal,
    )

    trace = receding_horizon_control(
        model,
        environment,
        goal_ids,
        goal_mask,
        CEMConfig(horizon=3),
        max_steps=4,
    )

    assert trace.reached_goal
    assert trace.positions.shape == (1, 2)
    assert trace.actions.shape == (0, 2)
    assert trace.planned_scores.shape == (0,)


def test_rollout_rejects_out_of_bounds_actions_before_learner_code() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        rollout_latent_candidates(
            ProvidedJointWAM(),
            torch.zeros(2),
            torch.tensor([[[1.1, 0.0]]]),
        )
