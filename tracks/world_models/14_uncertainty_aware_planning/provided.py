"""Provided WAM ensemble, planning problem, and validation for wm.14."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnsembleMixturePrediction:
    """Per-member mixture distributions over candidate future trajectories."""

    logits: torch.Tensor
    future_means: torch.Tensor


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Ensemble mean and decomposed candidate-level uncertainty."""

    mean_futures: torch.Tensor
    aleatoric: torch.Tensor
    epistemic: torch.Tensor


@dataclass(frozen=True)
class SelectedPlan:
    """Selected candidate and its nominal and uncertainty-aware metrics."""

    index: int
    name: str
    action_chunk: torch.Tensor
    mean_future: torch.Tensor
    nominal_score: torch.Tensor
    adjusted_score: torch.Tensor
    aleatoric: torch.Tensor
    epistemic: torch.Tensor


@dataclass(frozen=True)
class UncertaintyPlanningProblem:
    """Two plans designed to separate outcome spread from model disagreement."""

    candidate_names: tuple[str, ...]
    action_chunks: torch.Tensor
    goal_latent: torch.Tensor


class ProvidedWAMEnsemble:
    """Three frozen stochastic WAM proxies for a controlled comparison.

    Every member agrees that the familiar detour has two valid outcome modes.
    Members disagree about the unfamiliar shortcut: one predicts success, one
    predicts that progress stops early, and one predicts overshoot.
    """

    num_members = 3
    num_candidates = 2
    num_mixtures = 2
    horizon = 3
    state_dim = 2
    action_dim = 2

    @staticmethod
    def expected_action_chunks(
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the familiar detour and unfamiliar shortcut candidates."""

        return torch.tensor(
            [
                [[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            ],
            device=device,
            dtype=dtype,
        )

    def predict(self, action_chunks: torch.Tensor) -> EnsembleMixturePrediction:
        """Return [members, candidates, mixtures, horizon, state_dim] means."""

        expected_shape = (
            self.num_candidates,
            self.horizon,
            self.action_dim,
        )
        if action_chunks.shape != expected_shape:
            raise ValueError(
                "action_chunks must have shape [candidates, horizon, action_dim]"
            )
        if not torch.is_floating_point(action_chunks):
            raise ValueError("action_chunks must be floating point")
        if not torch.isfinite(action_chunks).all():
            raise ValueError("action_chunks must be finite")
        expected_actions = self.expected_action_chunks(
            action_chunks.device,
            action_chunks.dtype,
        )
        if not torch.allclose(action_chunks, expected_actions):
            raise ValueError("the controlled ensemble expects the provided candidates")

        left_future = action_chunks.new_tensor([[1.0, 0.0], [2.0, 0.5], [2.0, 0.5]])
        right_future = action_chunks.new_tensor([[1.0, 2.0], [2.0, 1.5], [2.0, 1.5]])
        familiar_modes = torch.stack((left_future, right_future))
        familiar_predictions = familiar_modes.unsqueeze(0).expand(
            self.num_members,
            -1,
            -1,
            -1,
        )

        shortcut_member_means = action_chunks.new_tensor(
            [
                [[1.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
                [[0.5, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[1.5, 1.0], [3.0, 1.0], [3.0, 1.0]],
            ]
        )
        shortcut_predictions = shortcut_member_means.unsqueeze(1).expand(
            -1,
            self.num_mixtures,
            -1,
            -1,
        )
        future_means = torch.stack(
            (familiar_predictions, shortcut_predictions),
            dim=1,
        )
        logits = action_chunks.new_zeros(
            self.num_members,
            self.num_candidates,
            self.num_mixtures,
        )
        return EnsembleMixturePrediction(logits, future_means)


def make_uncertainty_planning_problem() -> UncertaintyPlanningProblem:
    """Return two candidates with equal nominal mean futures but different risk."""

    return UncertaintyPlanningProblem(
        candidate_names=("familiar multimodal detour", "unfamiliar shortcut"),
        action_chunks=ProvidedWAMEnsemble.expected_action_chunks(),
        goal_latent=torch.tensor([2.0, 1.0]),
    )


def validate_ensemble_prediction(
    prediction: EnsembleMixturePrediction,
) -> None:
    """Validate ensemble mixture tensor shapes and values."""

    if prediction.logits.ndim != 3:
        raise ValueError("logits must have shape [members, candidates, mixtures]")
    if prediction.future_means.ndim != 5:
        raise ValueError(
            "future_means must have shape "
            "[members, candidates, mixtures, horizon, state_dim]"
        )
    if prediction.future_means.shape[:3] != prediction.logits.shape:
        raise ValueError("mixture logits and futures must align")
    if any(size == 0 for size in prediction.future_means.shape):
        raise ValueError("ensemble prediction dimensions must be positive")
    if prediction.logits.device != prediction.future_means.device:
        raise ValueError("ensemble prediction tensors must share a device")
    if prediction.logits.dtype != prediction.future_means.dtype:
        raise ValueError("ensemble prediction tensors must share a dtype")
    if not torch.is_floating_point(prediction.logits):
        raise ValueError("ensemble prediction tensors must be floating point")
    if (
        not torch.isfinite(prediction.logits).all()
        or not torch.isfinite(prediction.future_means).all()
    ):
        raise ValueError("ensemble prediction tensors must be finite")


def validate_uncertainty_inputs(
    prediction: EnsembleMixturePrediction,
) -> None:
    """Validate predictions before learner decomposition code runs."""

    validate_ensemble_prediction(prediction)
    if prediction.logits.shape[0] < 2:
        raise ValueError("uncertainty decomposition needs at least two members")


def validate_nominal_score_inputs(
    action_chunks: torch.Tensor,
    mean_futures: torch.Tensor,
    goal_latent: torch.Tensor,
    action_penalty: float,
) -> None:
    """Validate nominal candidate scoring tensors."""

    if action_chunks.ndim != 3:
        raise ValueError(
            "action_chunks must have shape [candidates, horizon, action_dim]"
        )
    if mean_futures.ndim != 3:
        raise ValueError(
            "mean_futures must have shape [candidates, horizon, state_dim]"
        )
    if action_chunks.shape[:2] != mean_futures.shape[:2]:
        raise ValueError("actions and mean futures must align in candidate and time")
    if goal_latent.shape != (mean_futures.shape[2],):
        raise ValueError("goal_latent must have shape [state_dim]")
    tensors = (action_chunks, mean_futures, goal_latent)
    if any(tensor.device != action_chunks.device for tensor in tensors):
        raise ValueError("nominal scoring tensors must share a device")
    if any(tensor.dtype != action_chunks.dtype for tensor in tensors):
        raise ValueError("nominal scoring tensors must share a dtype")
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("nominal scoring tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("nominal scoring tensors must be finite")
    if not math.isfinite(action_penalty) or action_penalty < 0.0:
        raise ValueError("action_penalty must be finite and non-negative")


def validate_risk_inputs(
    nominal_scores: torch.Tensor,
    epistemic_uncertainty: torch.Tensor,
    risk_coefficient: float,
) -> None:
    """Validate risk-adjustment inputs."""

    if nominal_scores.ndim != 1 or nominal_scores.shape[0] == 0:
        raise ValueError("nominal_scores must have shape [candidates]")
    if epistemic_uncertainty.shape != nominal_scores.shape:
        raise ValueError("epistemic_uncertainty must match nominal_scores")
    if nominal_scores.device != epistemic_uncertainty.device:
        raise ValueError("risk tensors must share a device")
    if nominal_scores.dtype != epistemic_uncertainty.dtype:
        raise ValueError("risk tensors must share a dtype")
    if not torch.is_floating_point(nominal_scores):
        raise ValueError("risk tensors must be floating point")
    if (
        not torch.isfinite(nominal_scores).all()
        or not torch.isfinite(epistemic_uncertainty).all()
    ):
        raise ValueError("risk tensors must be finite")
    if (epistemic_uncertainty < 0.0).any():
        raise ValueError("epistemic_uncertainty must be non-negative")
    if not math.isfinite(risk_coefficient) or risk_coefficient < 0.0:
        raise ValueError("risk_coefficient must be finite and non-negative")


def validate_selection_inputs(
    problem: UncertaintyPlanningProblem,
    estimate: UncertaintyEstimate,
    nominal_scores: torch.Tensor,
    adjusted_scores: torch.Tensor,
) -> None:
    """Validate aligned candidate fields before learner selection code."""

    candidate_count = problem.action_chunks.shape[0]
    if len(problem.candidate_names) != candidate_count:
        raise ValueError("candidate names must align with action chunks")
    if estimate.mean_futures.shape[:2] != problem.action_chunks.shape[:2]:
        raise ValueError("mean futures must align with candidate action chunks")
    expected_vector_shape = (candidate_count,)
    vectors = (
        estimate.aleatoric,
        estimate.epistemic,
        nominal_scores,
        adjusted_scores,
    )
    if any(vector.shape != expected_vector_shape for vector in vectors):
        raise ValueError("candidate metrics must have shape [candidates]")
    tensors = (estimate.mean_futures, *vectors)
    if any(tensor.device != problem.action_chunks.device for tensor in tensors):
        raise ValueError("candidate tensors must share a device")
    if any(tensor.dtype != problem.action_chunks.dtype for tensor in tensors):
        raise ValueError("candidate tensors must share a dtype")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("candidate tensors must be finite")
    if (estimate.aleatoric < 0.0).any() or (estimate.epistemic < 0.0).any():
        raise ValueError("uncertainty estimates must be non-negative")
