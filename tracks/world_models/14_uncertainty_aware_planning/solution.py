"""Reference solution for uncertainty-aware WAM planning."""

import torch

from provided import (
    EnsembleMixturePrediction,
    SelectedPlan,
    UncertaintyEstimate,
    UncertaintyPlanningProblem,
    validate_ensemble_prediction,
    validate_nominal_score_inputs,
    validate_risk_inputs,
    validate_selection_inputs,
    validate_uncertainty_inputs,
)


def mixture_expected_futures(
    prediction: EnsembleMixturePrediction,
) -> torch.Tensor:
    """Return per-member expectations as [members, candidates, horizon, state_dim]."""

    validate_ensemble_prediction(prediction)
    probabilities = torch.softmax(prediction.logits, dim=2)
    return (probabilities[..., None, None] * prediction.future_means).sum(dim=2)


def decompose_predictive_uncertainty(
    prediction: EnsembleMixturePrediction,
) -> UncertaintyEstimate:
    """Separate within-member multimodality from between-member disagreement."""

    validate_uncertainty_inputs(prediction)
    probabilities = torch.softmax(prediction.logits, dim=2)
    member_means = mixture_expected_futures(prediction)
    ensemble_mean = member_means.mean(dim=0)

    within_member_squared_deviation = (
        prediction.future_means - member_means.unsqueeze(2)
    ).square()
    member_aleatoric = (
        probabilities[..., None, None] * within_member_squared_deviation
    ).sum(dim=2)
    aleatoric = member_aleatoric.mean(dim=(0, 2, 3))

    between_member_squared_deviation = (
        member_means - ensemble_mean.unsqueeze(0)
    ).square()
    epistemic = between_member_squared_deviation.mean(dim=(0, 2, 3))
    return UncertaintyEstimate(ensemble_mean, aleatoric, epistemic)


def nominal_plan_scores(
    action_chunks: torch.Tensor,
    mean_futures: torch.Tensor,
    goal_latent: torch.Tensor,
    action_penalty: float = 0.05,
) -> torch.Tensor:
    """Score ensemble-mean goal accuracy and action effort; higher is better."""

    validate_nominal_score_inputs(
        action_chunks,
        mean_futures,
        goal_latent,
        action_penalty,
    )
    goal_cost = (mean_futures[:, -1] - goal_latent).square().sum(dim=1)
    action_cost = action_chunks.square().sum(dim=(1, 2))
    return -(goal_cost + action_penalty * action_cost)


def risk_adjusted_plan_scores(
    nominal_scores: torch.Tensor,
    epistemic_uncertainty: torch.Tensor,
    risk_coefficient: float,
) -> torch.Tensor:
    """Penalize epistemic model disagreement, not valid within-model modes."""

    validate_risk_inputs(
        nominal_scores,
        epistemic_uncertainty,
        risk_coefficient,
    )
    return nominal_scores - risk_coefficient * epistemic_uncertainty


def select_uncertainty_aware_plan(
    problem: UncertaintyPlanningProblem,
    estimate: UncertaintyEstimate,
    nominal_scores: torch.Tensor,
    adjusted_scores: torch.Tensor,
) -> SelectedPlan:
    """Select the candidate with the highest uncertainty-adjusted score."""

    validate_selection_inputs(
        problem,
        estimate,
        nominal_scores,
        adjusted_scores,
    )
    best_index = int(adjusted_scores.argmax().item())
    return SelectedPlan(
        index=best_index,
        name=problem.candidate_names[best_index],
        action_chunk=problem.action_chunks[best_index],
        mean_future=estimate.mean_futures[best_index],
        nominal_score=nominal_scores[best_index],
        adjusted_score=adjusted_scores[best_index],
        aleatoric=estimate.aleatoric[best_index],
        epistemic=estimate.epistemic[best_index],
    )
