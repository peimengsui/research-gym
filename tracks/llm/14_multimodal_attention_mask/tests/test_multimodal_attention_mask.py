import pytest
import torch
import torch.nn.functional as F

from implementation import (
    apply_attention_mask,
    apply_padding_mask,
    make_multimodal_attention_matrix,
    prefix_length,
    visual_prefix_causal_mask,
)


def test_prefix_length_includes_separator() -> None:
    assert prefix_length(4) == 5


def test_prefix_length_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="visual_token_count"):
        prefix_length(0)


def test_visual_prefix_causal_mask_matches_expected_pattern() -> None:
    # 2 visual + separator + 3 text = 6
    mask = visual_prefix_causal_mask(sequence_length=6, visual_token_count=2)

    expected = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_visual_prefix_is_bidirectional() -> None:
    mask = visual_prefix_causal_mask(sequence_length=5, visual_token_count=3)
    prefix = 4
    assert mask[:prefix, :prefix].all()
    assert not mask[:prefix, prefix:].any()


def test_text_region_is_causal_and_sees_prefix() -> None:
    mask = visual_prefix_causal_mask(sequence_length=7, visual_token_count=2)
    prefix = 3
    text = mask[prefix:]
    assert text[:, :prefix].all()
    assert torch.equal(text[:, prefix:], torch.ones(4, 4, dtype=torch.bool).tril())


def test_visual_prefix_causal_mask_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        visual_prefix_causal_mask(0, visual_token_count=2)
    with pytest.raises(ValueError, match="prefix"):
        visual_prefix_causal_mask(2, visual_token_count=2)


def test_apply_padding_mask_blocks_invalid_keys_and_queries() -> None:
    base = visual_prefix_causal_mask(sequence_length=6, visual_token_count=2)
    validity = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )

    matrix = apply_padding_mask(base, validity)

    assert matrix.shape == (2, 6, 6)
    assert matrix.dtype == torch.bool
    # Padding keys are never attended to.
    assert not matrix[0, :, 4:].any()
    assert not matrix[1, :, 5:].any()
    # Padding queries are all-False rows.
    assert not matrix[0, 4:].any()
    assert not matrix[1, 5:].any()
    # Real text still sees the full visual+separator prefix.
    assert matrix[0, 3, :3].all()
    assert matrix[1, 4, :3].all()


def test_apply_padding_mask_rejects_mismatched_shapes() -> None:
    base = torch.ones(4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="sequence length"):
        apply_padding_mask(base, torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        apply_padding_mask(torch.ones(4, 4), torch.ones(2, 4, dtype=torch.bool))


def test_make_multimodal_attention_matrix_end_to_end() -> None:
    validity = torch.tensor([[True, True, True, True, True, False]])
    matrix = make_multimodal_attention_matrix(validity, visual_token_count=2)

    expected = torch.tensor(
        [
            [
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, True, True, True, False],
                [False, False, False, False, False, False],
            ]
        ]
    )
    assert torch.equal(matrix, expected)


def test_apply_attention_mask_zeros_disallowed_weights() -> None:
    attention_matrix = make_multimodal_attention_matrix(
        torch.tensor([[True, True, True, True, False]]),
        visual_token_count=2,
    )
    scores = torch.ones(1, 5, 5)

    weights = apply_attention_mask(scores, attention_matrix)

    assert weights.shape == (1, 5, 5)
    assert torch.allclose(weights[0, :4].sum(dim=-1), torch.ones(4))
    assert torch.equal(weights[0, :4, 4], torch.zeros(4))
    # Uniform over the three prefix positions for the first visual query.
    assert torch.allclose(weights[0, 0, :3], torch.full((3,), 1.0 / 3.0))
    assert torch.equal(weights[0, 0, 3:], torch.zeros(2))
    # Padding query rows are all zeros, not NaN.
    assert torch.equal(weights[0, 4], torch.zeros(5))
    assert not torch.isnan(weights).any()


def test_apply_attention_mask_matches_manual_softmax() -> None:
    attention_matrix = visual_prefix_causal_mask(4, visual_token_count=1).unsqueeze(0)
    scores = torch.randn(1, 4, 4)

    weights = apply_attention_mask(scores, attention_matrix)

    manual = scores.masked_fill(~attention_matrix, float("-inf"))
    expected = F.softmax(manual, dim=-1)
    assert torch.allclose(weights, expected)


def test_masked_attention_is_invariant_to_disallowed_score_values() -> None:
    attention_matrix = make_multimodal_attention_matrix(
        torch.ones(1, 5, dtype=torch.bool),
        visual_token_count=2,
    )
    base_scores = torch.zeros(1, 5, 5)
    noisy_scores = base_scores.clone()
    noisy_scores[0, 0, 3] = 100.0  # future text for a visual query

    clean = apply_attention_mask(base_scores, attention_matrix)
    noisy = apply_attention_mask(noisy_scores, attention_matrix)

    assert torch.allclose(clean[0, 0], noisy[0, 0])


def test_batch_examples_can_differ_only_in_padding() -> None:
    validity = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )
    matrix = make_multimodal_attention_matrix(validity, visual_token_count=2)

    shared = visual_prefix_causal_mask(6, visual_token_count=2)
    assert torch.equal(matrix[1], shared)
    assert torch.equal(matrix[0, :4, :4], shared[:4, :4])
    assert not matrix[0, :, 4:].any()
