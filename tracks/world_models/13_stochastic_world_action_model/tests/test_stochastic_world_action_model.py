import torch

from implementation import (
    joint_strategy_mixture_nll,
    pack_strategy_targets,
    sample_coherent_hypotheses,
    score_strategy_hypotheses,
    select_best_strategy,
)
from provided import (
    JointMixturePrediction,
    ProvidedStochasticWAM,
    StrategyHypotheses,
    make_branching_problem,
    mixture_mean_strategy,
    strategy_hits_obstacle,
    unpack_strategy_vectors,
)


def make_prediction(
    component_log_scale: float = -3.0,
) -> tuple[
    ProvidedStochasticWAM,
    torch.Tensor,
    torch.Tensor,
    JointMixturePrediction,
]:
    model = ProvidedStochasticWAM(component_log_scale=component_log_scale)
    problem = make_branching_problem()
    current, goal, prediction = model(
        problem.current_videos,
        problem.goal_instruction_ids,
        problem.goal_instruction_mask,
    )
    return model, current, goal, prediction


def test_provided_wam_represents_two_routes_as_joint_components() -> None:
    model, _, _, prediction = make_prediction()
    actions, futures = unpack_strategy_vectors(
        prediction.means,
        model.horizon,
        model.action_dim,
        model.state_dim,
    )

    assert prediction.logits.shape == (1, 2)
    assert prediction.means.shape == prediction.log_scales.shape == (1, 2, 12)
    assert torch.equal(futures[0, 0, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(futures[0, 1, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(actions[0, 0, 0], torch.tensor([1.0, -1.0]))
    assert torch.equal(actions[0, 1, 0], torch.tensor([1.0, 1.0]))


def test_probability_weighted_mean_collides_but_components_do_not() -> None:
    model, _, _, prediction = make_prediction()
    problem = make_branching_problem()
    _, component_futures = unpack_strategy_vectors(
        prediction.means,
        model.horizon,
        model.action_dim,
        model.state_dim,
    )
    mean_strategy = mixture_mean_strategy(prediction)
    _, mean_futures = unpack_strategy_vectors(
        mean_strategy,
        model.horizon,
        model.action_dim,
        model.state_dim,
    )

    component_collisions = strategy_hits_obstacle(
        component_futures,
        problem.obstacle_latent,
        collision_radius=0.3,
    )
    mean_collision = strategy_hits_obstacle(
        mean_futures,
        problem.obstacle_latent,
        collision_radius=0.3,
    )

    assert not component_collisions.any()
    assert mean_collision.item()


def test_pack_strategy_targets_preserves_actions_then_futures() -> None:
    actions = torch.tensor([[[1.0, -1.0], [1.0, 1.0]]])
    futures = torch.tensor([[[1.0, 0.0], [2.0, 1.0]]])

    packed = pack_strategy_targets(actions, futures)

    assert packed.shape == (1, 8)
    assert torch.equal(
        packed,
        torch.tensor([[1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0]]),
    )


def test_joint_mixture_nll_prefers_a_coherent_mode_over_the_mean() -> None:
    model, _, _, prediction = make_prediction(component_log_scale=-2.0)
    component_actions, component_futures = unpack_strategy_vectors(
        prediction.means,
        model.horizon,
        model.action_dim,
        model.state_dim,
    )
    mean_actions, mean_futures = unpack_strategy_vectors(
        mixture_mean_strategy(prediction),
        model.horizon,
        model.action_dim,
        model.state_dim,
    )

    mode_loss = joint_strategy_mixture_nll(
        prediction,
        component_actions[:, 0],
        component_futures[:, 0],
    )
    mean_loss = joint_strategy_mixture_nll(
        prediction,
        mean_actions,
        mean_futures,
    )

    assert mode_loss < mean_loss


def test_joint_mixture_nll_has_finite_gradients() -> None:
    model, _, _, frozen_prediction = make_prediction(component_log_scale=-1.0)
    prediction = JointMixturePrediction(
        logits=frozen_prediction.logits.clone().requires_grad_(),
        means=frozen_prediction.means.clone().requires_grad_(),
        log_scales=frozen_prediction.log_scales.clone().requires_grad_(),
    )
    actions, futures = unpack_strategy_vectors(
        frozen_prediction.means[:, 0],
        model.horizon,
        model.action_dim,
        model.state_dim,
    )

    loss = joint_strategy_mixture_nll(prediction, actions, futures)
    loss.backward()

    assert loss.shape == ()
    for tensor in (prediction.logits, prediction.means, prediction.log_scales):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_sampling_uses_one_component_for_each_complete_strategy() -> None:
    model, _, _, prediction = make_prediction(component_log_scale=-20.0)

    hypotheses = sample_coherent_hypotheses(
        prediction,
        num_samples=64,
        horizon=model.horizon,
        action_dim=model.action_dim,
        state_dim=model.state_dim,
        generator=torch.Generator().manual_seed(5),
    )

    expected_first_columns = hypotheses.component_ids.to(torch.float32) * 2.0
    expected_action_columns = hypotheses.component_ids.to(torch.float32) * 2.0 - 1.0
    assert hypotheses.component_ids.shape == (1, 64)
    assert hypotheses.action_chunks.shape == (1, 64, 3, 2)
    assert hypotheses.future_latents.shape == (1, 64, 3, 2)
    assert torch.equal(hypotheses.component_ids.unique(), torch.tensor([0, 1]))
    assert torch.allclose(
        hypotheses.future_latents[0, :, 0, 1],
        expected_first_columns[0],
        atol=1e-5,
    )
    assert torch.allclose(
        hypotheses.action_chunks[0, :, 0, 1],
        expected_action_columns[0],
        atol=1e-5,
    )
    assert not hypotheses.action_chunks.requires_grad


def test_strategy_score_penalizes_collision_and_action_future_mismatch() -> None:
    current = torch.tensor([[0.0, 1.0]])
    goal = torch.tensor([[2.0, 1.0]])
    obstacle = torch.tensor([1.0, 1.0])
    safe_future = torch.tensor([[1.0, 0.0], [2.0, 1.0], [2.0, 1.0]])
    safe_actions = torch.tensor([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    straight_future = torch.tensor([[1.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
    straight_actions = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    mismatched_actions = torch.tensor([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])
    hypotheses = StrategyHypotheses(
        component_ids=torch.tensor([[0, 0, 1]]),
        action_chunks=torch.stack(
            (safe_actions, straight_actions, mismatched_actions)
        ).unsqueeze(0),
        future_latents=torch.stack(
            (safe_future, straight_future, safe_future)
        ).unsqueeze(0),
    )

    scores = score_strategy_hypotheses(
        hypotheses,
        current,
        goal,
        obstacle,
    )

    assert scores.shape == (1, 3)
    assert scores[0, 0] > scores[0, 1]
    assert scores[0, 0] > scores[0, 2]


def test_select_best_strategy_gathers_matching_fields_per_batch() -> None:
    hypotheses = StrategyHypotheses(
        component_ids=torch.tensor([[0, 1, 0], [1, 0, 1]]),
        action_chunks=torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
        future_latents=torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 2, 2),
    )
    scores = torch.tensor([[0.0, 3.0, 1.0], [4.0, 2.0, 1.0]])

    selected = select_best_strategy(hypotheses, scores)

    assert torch.equal(selected.component_ids, torch.tensor([1, 1]))
    assert torch.equal(selected.action_chunks[0], hypotheses.action_chunks[0, 1])
    assert torch.equal(selected.action_chunks[1], hypotheses.action_chunks[1, 0])
    assert torch.equal(selected.future_latents[0], hypotheses.future_latents[0, 1])
    assert torch.equal(selected.scores, torch.tensor([3.0, 4.0]))
