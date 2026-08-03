import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from implementation import (
    TinyGPT,
    apply_temperature,
    generate_tokens,
    sample_next_token,
    top_k_filter,
    top_p_filter,
)


def test_temperature_changes_distribution_sharpness() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])

    cold = F.softmax(apply_temperature(logits, 0.5), dim=-1)
    neutral = F.softmax(apply_temperature(logits, 1.0), dim=-1)
    hot = F.softmax(apply_temperature(logits, 2.0), dim=-1)

    assert cold[0, 0] > neutral[0, 0] > hot[0, 0]
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(logits, 0.0)


def test_top_k_keeps_exactly_k_logits() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])

    filtered = top_k_filter(logits, top_k=2)

    assert torch.equal(
        torch.isfinite(filtered), torch.tensor([[False, True, False, True]])
    )
    assert filtered[0, 1] == 4.0
    assert filtered[0, 3] == 3.0


def test_top_p_keeps_smallest_prefix_reaching_probability_mass() -> None:
    probabilities = torch.tensor([[0.50, 0.30, 0.15, 0.05]])
    logits = probabilities.log()

    filtered = top_p_filter(logits, top_p=0.70)

    assert torch.equal(
        torch.isfinite(filtered), torch.tensor([[True, True, False, False]])
    )
    assert torch.equal(top_p_filter(logits, 1.0), logits)


def test_token_selection_supports_greedy_and_restricted_sampling() -> None:
    logits = torch.tensor([[0.0, 1.0, 5.0, 2.0]])

    greedy = sample_next_token(logits, do_sample=False)
    sampled = sample_next_token(
        logits,
        top_k=1,
        generator=torch.Generator().manual_seed(10),
    )

    assert greedy.shape == (1, 1)
    assert greedy.item() == 2
    assert sampled.item() == 2


def test_tiny_gpt_produces_next_token_logits() -> None:
    model = TinyGPT(vocab_size=11, block_size=5, embed_dim=8, num_layers=1)
    token_ids = torch.randint(0, 11, (3, 5))

    logits, loss = model(token_ids)

    assert logits.shape == (3, 5, 11)
    assert loss is None


class IncrementModel(nn.Module):
    """Assign all probability to token (current + 1) modulo vocabulary size."""

    def __init__(self, vocab_size: int = 6, block_size: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.context_lengths: list[int] = []

    def forward(self, token_ids: torch.Tensor):
        self.context_lengths.append(token_ids.shape[1])
        next_ids = (token_ids + 1) % self.vocab_size
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            -math.inf,
            device=token_ids.device,
        )
        logits.scatter_(-1, next_ids.unsqueeze(-1), 0.0)
        return logits, None


def test_generation_preserves_prompt_and_crops_model_context() -> None:
    model = IncrementModel(block_size=3)
    prompt = torch.tensor([[0, 1, 2, 3]])

    generated = generate_tokens(
        model,
        prompt,
        max_new_tokens=4,
        top_k=1,
    )

    assert torch.equal(generated, torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]]))
    assert max(model.context_lengths) <= model.block_size


def test_generation_stops_at_end_of_sequence() -> None:
    model = IncrementModel()

    generated = generate_tokens(
        model,
        torch.tensor([[0]]),
        max_new_tokens=10,
        top_k=1,
        eos_token_id=3,
    )

    assert torch.equal(generated, torch.tensor([[0, 1, 2, 3]]))
