"""Learner scaffold for uncertainty-aware WAM planning."""

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
    """Return per-member expectations as [members, candidates, horizon, state_dim].

    prediction.logits: [members, candidates, mixtures]
    prediction.future_means: [members, candidates, mixtures, horizon, state_dim]
    """

    validate_ensemble_prediction(prediction)
    # TODO 1: softmax logits across mixture components. Add horizon and state
    # singleton dimensions to the probabilities, multiply by component future
    # means, and sum over mixtures. Keep members and candidates separate.
    raise NotImplementedError("TODO: compute each member's expected futures")


def decompose_predictive_uncertainty(
    prediction: EnsembleMixturePrediction,
) -> UncertaintyEstimate:
    """Separate within-member multimodality from between-member disagreement.

    returns mean_futures: [candidates, horizon, state_dim]
    returns aleatoric: [candidates]
    returns epistemic: [candidates]
    """

    validate_uncertainty_inputs(prediction)
    # TODO 2: compute per-member expected futures, then their ensemble mean.
    # Aleatoric uncertainty is the mixture-probability-weighted squared distance
    # from component futures to their own member mean, averaged over members,
    # horizon, and state. Epistemic uncertainty is squared distance from each
    # member mean to the ensemble mean, averaged over members, horizon, and state.
    # Use population means rather than an unbiased sample variance.
    raise NotImplementedError("TODO: decompose predictive uncertainty")


def nominal_plan_scores(
    action_chunks: torch.Tensor,
    mean_futures: torch.Tensor,
    goal_latent: torch.Tensor,
    action_penalty: float = 0.05,
) -> torch.Tensor:
    """Score ensemble-mean goal accuracy and action effort; higher is better.

    action_chunks: [candidates, horizon, action_dim]
    mean_futures: [candidates, horizon, state_dim]
    goal_latent: [state_dim]
    returns: [candidates]
    """

    validate_nominal_score_inputs(
        action_chunks,
        mean_futures,
        goal_latent,
        action_penalty,
    )
    # TODO 3: add final squared goal distance and action_penalty times total
    # squared action effort for each candidate. Negate the cost so higher scores
    # are better. This intentionally ignores uncertainty to create a baseline.
    raise NotImplementedError("TODO: score nominal candidate utility")


def risk_adjusted_plan_scores(
    nominal_scores: torch.Tensor,
    epistemic_uncertainty: torch.Tensor,
    risk_coefficient: float,
) -> torch.Tensor:
    """Penalize epistemic model disagreement, not valid within-model modes.

    nominal_scores: [candidates]
    epistemic_uncertainty: [candidates]
    returns: [candidates]
    """

    validate_risk_inputs(
        nominal_scores,
        epistemic_uncertainty,
        risk_coefficient,
    )
    # TODO 4: subtract risk_coefficient * epistemic_uncertainty. Do not include
    # aleatoric uncertainty: multiple agreed-upon modes can all be valid.
    raise NotImplementedError("TODO: compute epistemic-risk-adjusted scores")


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
    # TODO 5: find the highest adjusted score, convert its scalar index to a
    # Python int, and return the matching name, action chunk, ensemble-mean
    # future, nominal/adjusted scores, and both uncertainty values.
    raise NotImplementedError("TODO: select the uncertainty-aware plan")
