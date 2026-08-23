"""Tiny probability models supplied for the speculative decoding lesson."""

import torch


class BigramLanguageModel:
    """Return a next-token distribution from the final context token."""

    def __init__(self, transition_probabilities: torch.Tensor):
        if (
            transition_probabilities.ndim != 2
            or transition_probabilities.shape[0] != transition_probabilities.shape[1]
            or not transition_probabilities.is_floating_point()
        ):
            raise ValueError("transition_probabilities must be square and floating")
        if (transition_probabilities < 0).any():
            raise ValueError("transition probabilities must be non-negative")
        row_sums = transition_probabilities.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums)):
            raise ValueError("each transition row must sum to one")
        self.transition_probabilities = transition_probabilities.clone()
        self.vocab_size = transition_probabilities.shape[0]
        self.verification_calls = 0

    def next_probabilities(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 1 or context.dtype != torch.long or context.numel() == 0:
            raise ValueError("context must be a non-empty 1D torch.long tensor")
        return self.transition_probabilities[context[-1]]

    def score_draft(
        self,
        prefix: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Score every draft position plus one bonus position in one call."""

        if draft_tokens.ndim != 1 or draft_tokens.dtype != torch.long:
            raise ValueError("draft_tokens must be a 1D torch.long tensor")
        self.verification_calls += 1
        context = prefix.clone()
        distributions = []
        for token in draft_tokens:
            distributions.append(self.next_probabilities(context))
            context = torch.cat((context, token.reshape(1)))
        distributions.append(self.next_probabilities(context))
        return torch.stack(distributions)

    def reset_verification_calls(self) -> None:
        self.verification_calls = 0


def make_toy_speculative_models() -> tuple[BigramLanguageModel, BigramLanguageModel]:
    """Return a target and an imperfect but similar draft model."""

    target = torch.tensor(
        [
            [0.05, 0.55, 0.25, 0.10, 0.05],
            [0.05, 0.10, 0.55, 0.25, 0.05],
            [0.05, 0.10, 0.10, 0.65, 0.10],
            [0.05, 0.10, 0.10, 0.15, 0.60],
            [0.05, 0.05, 0.05, 0.05, 0.80],
        ]
    )
    draft = torch.tensor(
        [
            [0.05, 0.60, 0.20, 0.10, 0.05],
            [0.05, 0.10, 0.60, 0.20, 0.05],
            [0.05, 0.10, 0.10, 0.60, 0.15],
            [0.05, 0.10, 0.10, 0.20, 0.55],
            [0.05, 0.05, 0.05, 0.05, 0.80],
        ]
    )
    return BigramLanguageModel(target), BigramLanguageModel(draft)
