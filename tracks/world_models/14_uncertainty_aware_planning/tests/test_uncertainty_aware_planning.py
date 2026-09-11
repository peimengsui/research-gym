import pytest
import torch

from implementation import (
    decompose_predictive_uncertainty,
    mixture_expected_futures,
    nominal_plan_scores,
    risk_adjusted_plan_scores,
    select_uncertainty_aware_plan,
)
from provided import (
    EnsembleMixturePrediction,
    ProvidedWAMEnsemble,
    make_uncertainty_planning_problem,
)


def make_prediction() -> tuple[
    ProvidedWAMEnsemble,
    EnsembleMixturePrediction,
]:
    ensemble = ProvidedWAMEnsemble()
    problem = make_uncertainty_planning_problem()
    return ensemble, ensemble.predict(problem.action_chunks)


def test_provided_ensemble_returns_member_candidate_mixture_futures() -> None:
    ensemble, prediction = make_prediction()

    assert prediction.logits.shape == (3, 2, 2)
    assert prediction.future_means.shape == (3, 2, 2, 3, 2)
    assert torch.equal(
        prediction.future_means[0, 0, 0],
        torch.tensor([[1.0, 0.0], [2.0, 0.5], [2.0, 0.5]]),
    )
    assert torch.equal(
        prediction.future_means[1, 1, 0],
        torch.tensor([[0.5, 1.0], [1.0, 1.0], [1.0, 1.0]]),
    )
    assert ensemble.num_members == 3


def test_mixture_expectation_keeps_members_and_candidates_separate() -> None:
    _, prediction = make_prediction()

    member_means = mixture_expected_futures(prediction)

    expected_shared_mean = torch.tensor([[1.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
    assert member_means.shape == (3, 2, 3, 2)
    assert torch.equal(member_means[:, 0], expected_shared_mean.expand(3, -1, -1))
    assert torch.equal(member_means[0, 1], expected_shared_mean)
    assert not torch.equal(member_means[1, 1], member_means[2, 1])


def test_uncertainty_decomposition_separates_modes_from_disagreement() -> None:
    _, prediction = make_prediction()

    estimate = decompose_predictive_uncertainty(prediction)

    assert estimate.mean_futures.shape == (2, 3, 2)
    assert torch.allclose(estimate.mean_futures[0], estimate.mean_futures[1])
    assert torch.allclose(estimate.aleatoric, torch.tensor([0.25, 0.0]))
    assert torch.allclose(estimate.epistemic, torch.tensor([0.0, 0.25]))


def test_decomposition_is_unchanged_by_component_order() -> None:
    _, prediction = make_prediction()
    reversed_prediction = EnsembleMixturePrediction(
        logits=prediction.logits.flip(dims=(2,)),
        future_means=prediction.future_means.flip(dims=(2,)),
    )

    original = decompose_predictive_uncertainty(prediction)
    reversed_estimate = decompose_predictive_uncertainty(reversed_prediction)

    assert torch.allclose(original.mean_futures, reversed_estimate.mean_futures)
    assert torch.allclose(original.aleatoric, reversed_estimate.aleatoric)
    assert torch.allclose(original.epistemic, reversed_estimate.epistemic)


def test_nominal_score_prefers_lower_effort_shortcut() -> None:
    problem = make_uncertainty_planning_problem()
    _, prediction = make_prediction()
    estimate = decompose_predictive_uncertainty(prediction)

    scores = nominal_plan_scores(
        problem.action_chunks,
        estimate.mean_futures,
        problem.goal_latent,
    )

    assert torch.allclose(scores, torch.tensor([-0.2, -0.1]))
    assert scores.argmax().item() == 1


def test_risk_adjustment_penalizes_epistemic_not_aleatoric_uncertainty() -> None:
    nominal = torch.tensor([-0.2, -0.1])
    epistemic = torch.tensor([0.0, 0.25])

    adjusted = risk_adjusted_plan_scores(
        nominal,
        epistemic,
        risk_coefficient=1.0,
    )

    assert torch.allclose(adjusted, torch.tensor([-0.2, -0.35]))
    assert adjusted.argmax().item() == 0


def test_selection_changes_from_shortcut_to_familiar_detour_with_risk() -> None:
    problem = make_uncertainty_planning_problem()
    _, prediction = make_prediction()
    estimate = decompose_predictive_uncertainty(prediction)
    nominal = nominal_plan_scores(
        problem.action_chunks,
        estimate.mean_futures,
        problem.goal_latent,
    )
    risk_neutral = select_uncertainty_aware_plan(
        problem,
        estimate,
        nominal,
        nominal,
    )
    adjusted = risk_adjusted_plan_scores(nominal, estimate.epistemic, 1.0)
    risk_aware = select_uncertainty_aware_plan(
        problem,
        estimate,
        nominal,
        adjusted,
    )

    assert risk_neutral.index == 1
    assert risk_neutral.name == "unfamiliar shortcut"
    assert risk_aware.index == 0
    assert risk_aware.name == "familiar multimodal detour"
    assert torch.equal(risk_aware.action_chunk, problem.action_chunks[0])
    assert risk_aware.aleatoric > risk_aware.epistemic


def test_decomposition_rejects_single_member_ensemble_before_learner_code() -> None:
    _, prediction = make_prediction()
    single_member = EnsembleMixturePrediction(
        logits=prediction.logits[:1],
        future_means=prediction.future_means[:1],
    )

    with pytest.raises(ValueError, match="at least two"):
        decompose_predictive_uncertainty(single_member)
