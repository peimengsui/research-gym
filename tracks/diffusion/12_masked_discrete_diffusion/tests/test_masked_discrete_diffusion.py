import math

import torch

from implementation import (
    coupled_mask_trajectory,
    eligible_token_mask,
    linear_mask_schedule,
    masked_token_cross_entropy,
    q_sample_tokens,
)
from provided import make_toy_token_batch


def test_linear_schedule_has_clean_and_fully_masked_endpoints() -> None:
    schedule = linear_mask_schedule(5)

    assert torch.equal(schedule, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_eligibility_excludes_padding_special_and_existing_mask_tokens() -> None:
    input_ids = torch.tensor([[1, 3, 10, 2, 0]])
    attention_mask = torch.tensor([[True, True, True, True, False]])
    special_token_mask = torch.tensor([[True, False, False, True, False]])

    eligible = eligible_token_mask(
        input_ids,
        attention_mask,
        special_token_mask,
        mask_token_id=10,
    )

    assert torch.equal(
        eligible,
        torch.tensor([[False, True, False, False, False]]),
    )


def test_direct_corruption_has_exact_clean_and_masked_endpoints() -> None:
    tokens = make_toy_token_batch()
    schedule = linear_mask_schedule(5)
    clean_ids = tokens.input_ids.clone()

    batch = q_sample_tokens(
        tokens.input_ids,
        torch.tensor([0, 2, 4]),
        schedule,
        tokens.mask_token_id,
        tokens.attention_mask,
        tokens.special_token_mask,
        uniforms=torch.full(tokens.input_ids.shape, 0.4),
    )

    assert torch.equal(batch.corrupted_ids[0], tokens.input_ids[0])
    assert torch.equal(
        batch.corruption_mask[1],
        batch.eligible_mask[1],
    )
    assert torch.equal(
        batch.corruption_mask[2],
        batch.eligible_mask[2],
    )
    assert torch.equal(batch.loss_mask, batch.corruption_mask)
    assert torch.equal(tokens.input_ids, clean_ids)


def test_each_row_uses_its_own_timestep_probability() -> None:
    input_ids = torch.tensor([[1, 3, 4, 2], [1, 3, 4, 2]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    special_mask = (input_ids == 1) | (input_ids == 2)
    schedule = linear_mask_schedule(3)
    uniforms = torch.tensor([[0.0, 0.4, 0.6, 0.0], [0.0, 0.4, 0.6, 0.0]])

    batch = q_sample_tokens(
        input_ids,
        torch.tensor([1, 2]),
        schedule,
        10,
        attention_mask,
        special_mask,
        uniforms,
    )

    assert torch.equal(
        batch.corruption_mask[0], torch.tensor([False, True, False, False])
    )
    assert torch.equal(
        batch.corruption_mask[1], torch.tensor([False, True, True, False])
    )
    assert torch.allclose(batch.masking_probabilities, torch.tensor([0.5, 1.0]))


def test_masked_cross_entropy_only_has_gradients_at_loss_positions() -> None:
    logits = torch.zeros(1, 4, 6, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4]])
    loss_mask = torch.tensor([[False, True, False, True]])

    loss = masked_token_cross_entropy(logits, targets, loss_mask)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(math.log(6.0)))
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 0], torch.zeros(6))
    assert torch.equal(logits.grad[0, 2], torch.zeros(6))
    assert torch.count_nonzero(logits.grad[0, 1]) > 0
    assert torch.count_nonzero(logits.grad[0, 3]) > 0


def test_empty_loss_mask_returns_graph_connected_zero() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.zeros(2, 3, dtype=torch.long)
    loss_mask = torch.zeros(2, 3, dtype=torch.bool)

    loss = masked_token_cross_entropy(logits, targets, loss_mask)
    loss.backward()

    assert loss == 0.0
    assert loss.requires_grad
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_coupled_trajectory_is_absorbing_and_preserves_structure() -> None:
    tokens = make_toy_token_batch()
    schedule = linear_mask_schedule(5)
    uniforms = torch.tensor(
        [
            [0.9, 0.1, 0.4, 0.7, 0.2, 0.8, 0.3],
            [0.6, 0.2, 0.8, 0.1, 0.5, 0.4, 0.3],
            [0.7, 0.6, 0.3, 0.9, 0.1, 0.8, 0.2],
        ]
    )

    trajectory = coupled_mask_trajectory(
        tokens.input_ids,
        schedule,
        tokens.mask_token_id,
        tokens.attention_mask,
        tokens.special_token_mask,
        uniforms,
    )

    assert trajectory.shape == (5, 3, 7)
    assert torch.equal(trajectory[0], tokens.input_ids)
    eligible = eligible_token_mask(
        tokens.input_ids,
        tokens.attention_mask,
        tokens.special_token_mask,
        tokens.mask_token_id,
    )
    assert torch.all(trajectory[-1][eligible] == tokens.mask_token_id)
    for previous, current in zip(trajectory[:-1], trajectory[1:]):
        previously_masked = previous == tokens.mask_token_id
        assert torch.all(current[previously_masked] == tokens.mask_token_id)
    protected = ~eligible
    for state in trajectory:
        assert torch.equal(state[protected], tokens.input_ids[protected])


def test_seeded_corruption_is_reproducible() -> None:
    tokens = make_toy_token_batch()
    schedule = linear_mask_schedule(5)
    timesteps = torch.full((3,), 2)

    first = q_sample_tokens(
        tokens.input_ids,
        timesteps,
        schedule,
        tokens.mask_token_id,
        tokens.attention_mask,
        tokens.special_token_mask,
        generator=torch.Generator().manual_seed(12),
    )
    second = q_sample_tokens(
        tokens.input_ids,
        timesteps,
        schedule,
        tokens.mask_token_id,
        tokens.attention_mask,
        tokens.special_token_mask,
        generator=torch.Generator().manual_seed(12),
    )

    assert torch.equal(first.corrupted_ids, second.corrupted_ids)
