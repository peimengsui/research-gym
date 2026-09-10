"""Reference solution for coherent stochastic world-action strategies."""

import math

import torch
import torch.nn.functional as F

from provided import (
    JointMixturePrediction,
    SelectedStrategy,
    StrategyHypotheses,
    strategy_hits_obstacle,
    unpack_strategy_vectors,
    validate_loss_inputs,
    validate_pack_inputs,
    validate_sampling_inputs,
    validate_scoring_inputs,
    validate_selection_inputs,
)


def pack_strategy_targets(
    action_chunks: torch.Tensor,
    future_latents: torch.Tensor,
) -> torch.Tensor:
    """Pack aligned targets as [batch, joint_strategy_dim]."""

    validate_pack_inputs(action_chunks, future_latents)
    return torch.cat(
        (
            action_chunks.flatten(start_dim=1),
            future_latents.flatten(start_dim=1),
        ),
        dim=1,
    )


def joint_strategy_mixture_nll(
    prediction: JointMixturePrediction,
    action_chunks: torch.Tensor,
    future_latents: torch.Tensor,
) -> torch.Tensor:
    """Return NLL under a mixture over complete action-and-future strategies."""

    validate_loss_inputs(prediction, action_chunks, future_latents)
    targets = pack_strategy_targets(action_chunks, future_latents).unsqueeze(1)
    normalized_errors = (targets - prediction.means) * torch.exp(-prediction.log_scales)
    component_log_probabilities = -0.5 * (
        normalized_errors.square()
        + 2.0 * prediction.log_scales
        + math.log(2.0 * math.pi)
    ).sum(dim=2)
    mixture_log_probabilities = (
        F.log_softmax(prediction.logits, dim=1) + component_log_probabilities
    )
    return -torch.logsumexp(mixture_log_probabilities, dim=1).mean()


@torch.no_grad()
def sample_coherent_hypotheses(
    prediction: JointMixturePrediction,
    num_samples: int,
    horizon: int,
    action_dim: int,
    state_dim: int,
    generator: torch.Generator | None = None,
) -> StrategyHypotheses:
    """Sample one component id for every complete strategy hypothesis."""

    validate_sampling_inputs(
        prediction,
        num_samples,
        horizon,
        action_dim,
        state_dim,
    )
    probabilities = torch.softmax(prediction.logits, dim=1)
    component_ids = torch.multinomial(
        probabilities,
        num_samples,
        replacement=True,
        generator=generator,
    )
    gather_indices = component_ids.unsqueeze(2).expand(
        -1,
        -1,
        prediction.means.shape[2],
    )
    selected_means = prediction.means.gather(1, gather_indices)
    selected_log_scales = prediction.log_scales.gather(1, gather_indices)
    noise = torch.randn(
        selected_means.shape,
        device=selected_means.device,
        dtype=selected_means.dtype,
        generator=generator,
    )
    sampled_vectors = selected_means + selected_log_scales.exp() * noise
    action_chunks, future_latents = unpack_strategy_vectors(
        sampled_vectors,
        horizon,
        action_dim,
        state_dim,
    )
    return StrategyHypotheses(component_ids, action_chunks, future_latents)


def score_strategy_hypotheses(
    hypotheses: StrategyHypotheses,
    current_latents: torch.Tensor,
    goal_latents: torch.Tensor,
    obstacle_latent: torch.Tensor,
    collision_radius: float = 0.3,
    collision_penalty: float = 10.0,
    coherence_penalty: float = 1.0,
    action_penalty: float = 0.01,
) -> torch.Tensor:
    """Score goal accuracy, safety, action/future coherence, and effort."""

    validate_scoring_inputs(
        hypotheses,
        current_latents,
        goal_latents,
        obstacle_latent,
        collision_radius,
        collision_penalty,
        coherence_penalty,
        action_penalty,
    )
    sample_count = hypotheses.component_ids.shape[1]
    initial_latents = current_latents[:, None, None, :].expand(
        -1,
        sample_count,
        1,
        -1,
    )
    previous_latents = torch.cat(
        (initial_latents, hypotheses.future_latents[:, :, :-1]),
        dim=2,
    )
    implied_actions = hypotheses.future_latents - previous_latents
    coherence_cost = (
        (hypotheses.action_chunks - implied_actions).square().sum(dim=(2, 3))
    )
    goal_cost = (
        (hypotheses.future_latents[:, :, -1] - goal_latents[:, None])
        .square()
        .sum(dim=2)
    )
    collision_cost = strategy_hits_obstacle(
        hypotheses.future_latents,
        obstacle_latent,
        collision_radius,
    ).to(hypotheses.action_chunks.dtype)
    action_cost = hypotheses.action_chunks.square().sum(dim=(2, 3))
    return -(
        goal_cost
        + collision_penalty * collision_cost
        + coherence_penalty * coherence_cost
        + action_penalty * action_cost
    )


def select_best_strategy(
    hypotheses: StrategyHypotheses,
    scores: torch.Tensor,
) -> SelectedStrategy:
    """Gather the highest-scoring sampled strategy for every batch item."""

    validate_selection_inputs(hypotheses, scores)
    best_indices = scores.argmax(dim=1)
    batch_indices = torch.arange(
        scores.shape[0],
        device=scores.device,
    )
    return SelectedStrategy(
        component_ids=hypotheses.component_ids[batch_indices, best_indices],
        action_chunks=hypotheses.action_chunks[batch_indices, best_indices],
        future_latents=hypotheses.future_latents[batch_indices, best_indices],
        scores=scores[batch_indices, best_indices],
    )
