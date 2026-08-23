"""Learner scaffold for exact speculative decoding.

Tiny target and draft probability models are complete in provided.py. TODOs
focus on acceptance, correction, block drafting, verification, and generation.
"""

from dataclasses import dataclass

import torch

from provided import BigramLanguageModel


def sample_token(
    probabilities: torch.Tensor,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token ID from a one-dimensional probability vector."""

    if probabilities.ndim != 1 or not probabilities.is_floating_point():
        raise ValueError("probabilities must be a 1D floating-point tensor")
    if (probabilities < 0).any() or probabilities.sum() <= 0:
        raise ValueError("probabilities must be non-negative with positive mass")
    normalized = probabilities / probabilities.sum()
    return int(torch.multinomial(normalized, 1, generator=generator).item())


def draft_acceptance_probability(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
    draft_token_id: int,
) -> torch.Tensor:
    """Return min(1, p(token) / q(token)) for one drafted token."""

    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError("target and draft distributions must have matching shapes")
    if not 0 <= draft_token_id < target_probabilities.numel():
        raise ValueError("draft_token_id is outside the vocabulary")
    # TODO 1: Read p and q at draft_token_id. Reject a zero q probability, then
    # return the scalar tensor min(1, p / q).
    raise NotImplementedError


def rejection_correction_distribution(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return normalized max(p - q, 0) used after a draft rejection."""

    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError("target and draft distributions must have matching shapes")
    # TODO 2: Clamp p - q at zero, require positive remaining mass, and normalize
    # the residual into a probability distribution. This correction—not p alone—
    # makes the final output distribution match the target exactly.
    raise NotImplementedError


def sample_draft_block(
    draft_model: BigramLanguageModel,
    prefix: torch.Tensor,
    draft_length: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a draft block and retain q distributions for verification."""

    if draft_length <= 0:
        raise ValueError("draft_length must be positive")
    # TODO 3: Autoregressively sample draft_length tokens from the draft model.
    # Extend a local 1D context after each token and retain each q distribution.
    # Return tokens shaped (draft_length,) and distributions shaped
    # (draft_length, vocab_size).
    raise NotImplementedError


@dataclass(frozen=True)
class SpeculativeStepResult:
    tokens: torch.Tensor
    drafted_tokens: int
    accepted_draft_tokens: int
    rejected: bool


def speculative_step(
    target_model: BigramLanguageModel,
    draft_model: BigramLanguageModel,
    prefix: torch.Tensor,
    draft_length: int,
    generator: torch.Generator | None = None,
) -> SpeculativeStepResult:
    """Draft, verify once with the target, and emit one or more exact samples."""

    # TODO 4:
    # 1. Sample a draft block and call target_model.score_draft exactly once.
    #    It returns p for every drafted position plus one bonus position.
    # 2. For each draft token, draw uniform random noise and accept when it is
    #    <= min(1, p(token) / q(token)). Append every accepted token.
    # 3. At the first rejection, sample one token from normalized max(p-q, 0),
    #    append it, and return with rejected=True.
    # 4. If all drafts are accepted, sample and append one target bonus token.
    raise NotImplementedError


@dataclass(frozen=True)
class SpeculativeGenerationResult:
    token_ids: torch.Tensor
    drafted_tokens: int
    accepted_draft_tokens: int
    target_verification_calls: int

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.drafted_tokens


def speculative_generate(
    target_model: BigramLanguageModel,
    draft_model: BigramLanguageModel,
    prompt: torch.Tensor,
    max_new_tokens: int,
    draft_length: int,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> SpeculativeGenerationResult:
    """Generate exactly from the target distribution using speculative blocks."""

    if prompt.ndim != 1 or prompt.dtype != torch.long or prompt.numel() == 0:
        raise ValueError("prompt must be a non-empty 1D torch.long tensor")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if draft_length <= 0:
        raise ValueError("draft_length must be positive")
    if eos_token_id is not None and not 0 <= eos_token_id < target_model.vocab_size:
        raise ValueError("eos_token_id is outside the vocabulary")

    # TODO 5: Repeatedly call speculative_step, append emitted tokens without
    # exceeding max_new_tokens, stop at EOS, and accumulate drafted, accepted,
    # and target-call counts. Return the original prompt plus generated tokens.
    raise NotImplementedError
