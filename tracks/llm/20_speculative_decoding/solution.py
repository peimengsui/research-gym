"""Reference solution for exact speculative decoding."""

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
    draft_probability = draft_probabilities[draft_token_id]
    if draft_probability <= 0:
        raise ValueError("the drafted token must have positive draft probability")
    return torch.clamp(
        target_probabilities[draft_token_id] / draft_probability,
        max=1.0,
    )


def rejection_correction_distribution(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return normalized max(p - q, 0) used after a draft rejection."""

    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError("target and draft distributions must have matching shapes")
    residual = torch.clamp(target_probabilities - draft_probabilities, min=0.0)
    if residual.sum() <= 0:
        raise ValueError("correction distribution requires positive residual mass")
    return residual / residual.sum()


def sample_draft_block(
    draft_model: BigramLanguageModel,
    prefix: torch.Tensor,
    draft_length: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a draft block and retain q distributions for verification."""

    if draft_length <= 0:
        raise ValueError("draft_length must be positive")
    context = prefix.clone()
    tokens = []
    distributions = []
    for _ in range(draft_length):
        probabilities = draft_model.next_probabilities(context)
        token = sample_token(probabilities, generator)
        tokens.append(token)
        distributions.append(probabilities)
        context = torch.cat((context, torch.tensor([token], dtype=torch.long)))
    return torch.tensor(tokens, dtype=torch.long), torch.stack(distributions)


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

    draft_tokens, draft_distributions = sample_draft_block(
        draft_model, prefix, draft_length, generator
    )
    target_distributions = target_model.score_draft(prefix, draft_tokens)
    emitted = []
    for index, draft_token in enumerate(draft_tokens.tolist()):
        acceptance = draft_acceptance_probability(
            target_distributions[index], draft_distributions[index], draft_token
        )
        uniform = torch.rand((), generator=generator)
        if uniform <= acceptance:
            emitted.append(draft_token)
            continue
        correction = rejection_correction_distribution(
            target_distributions[index], draft_distributions[index]
        )
        emitted.append(sample_token(correction, generator))
        return SpeculativeStepResult(
            tokens=torch.tensor(emitted, dtype=torch.long),
            drafted_tokens=draft_length,
            accepted_draft_tokens=index,
            rejected=True,
        )

    emitted.append(sample_token(target_distributions[-1], generator))
    return SpeculativeStepResult(
        tokens=torch.tensor(emitted, dtype=torch.long),
        drafted_tokens=draft_length,
        accepted_draft_tokens=draft_length,
        rejected=False,
    )


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

    generated = prompt.clone()
    drafted = 0
    accepted = 0
    target_calls = 0
    while generated.numel() - prompt.numel() < max_new_tokens:
        step = speculative_step(
            target_model, draft_model, generated, draft_length, generator
        )
        drafted += step.drafted_tokens
        accepted += step.accepted_draft_tokens
        target_calls += 1
        for token in step.tokens.tolist():
            if generated.numel() - prompt.numel() == max_new_tokens:
                break
            generated = torch.cat((generated, torch.tensor([token], dtype=torch.long)))
            if eos_token_id is not None and token == eos_token_id:
                return SpeculativeGenerationResult(
                    generated, drafted, accepted, target_calls
                )
    return SpeculativeGenerationResult(generated, drafted, accepted, target_calls)
