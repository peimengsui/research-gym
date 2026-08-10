import pytest
import torch

from implementation import (
    SEPARATOR_TOKEN_TYPE,
    TEXT_TOKEN_TYPE,
    VISUAL_TOKEN_TYPE,
    MultimodalSequenceEmbedding,
    make_multimodal_attention_mask,
    make_text_padding_mask,
    make_token_type_ids,
)
from provided import TinyVisionEncoder


def make_vision_encoder(embed_dim: int = 8) -> TinyVisionEncoder:
    return TinyVisionEncoder(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        embed_dim=embed_dim,
        num_heads=2,
        num_layers=1,
    )


def make_model(embed_dim: int = 8) -> MultimodalSequenceEmbedding:
    return MultimodalSequenceEmbedding(
        vision_encoder=make_vision_encoder(embed_dim),
        vocab_size=16,
        max_text_tokens=5,
        embed_dim=embed_dim,
    )


def test_text_padding_mask_marks_only_real_tokens() -> None:
    token_ids = torch.tensor([[4, 5, 0, 0], [7, 0, 8, 0]])

    mask = make_text_padding_mask(token_ids, pad_token_id=0)

    assert mask.dtype == torch.bool
    assert torch.equal(
        mask,
        torch.tensor([[True, True, False, False], [True, False, True, False]]),
    )


def test_text_padding_mask_rejects_non_batched_ids() -> None:
    with pytest.raises(ValueError, match="text_token_ids"):
        make_text_padding_mask(torch.tensor([1, 2, 0]), pad_token_id=0)


def test_multimodal_mask_keeps_visual_separator_and_real_text() -> None:
    text_mask = torch.tensor([[True, False, False], [True, True, False]])

    mask = make_multimodal_attention_mask(text_mask, num_visual_tokens=4)

    assert mask.shape == (2, 8)
    assert mask[:, :5].all()
    assert torch.equal(mask[:, 5:], text_mask)


@pytest.mark.parametrize(
    ("text_mask", "num_visual_tokens"),
    [(torch.ones(2, 3), 4), (torch.ones(2, 3, dtype=torch.bool), 0)],
)
def test_multimodal_mask_rejects_invalid_inputs(
    text_mask: torch.Tensor,
    num_visual_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        make_multimodal_attention_mask(text_mask, num_visual_tokens)


def test_token_type_ids_describe_each_sequence_region() -> None:
    token_ids = torch.tensor([[4, 5, 0], [7, 8, 9]])

    type_ids = make_token_type_ids(token_ids, num_visual_tokens=2)

    expected = torch.tensor(
        [
            [VISUAL_TOKEN_TYPE, VISUAL_TOKEN_TYPE, SEPARATOR_TOKEN_TYPE]
            + [TEXT_TOKEN_TYPE] * 3,
            [VISUAL_TOKEN_TYPE, VISUAL_TOKEN_TYPE, SEPARATOR_TOKEN_TYPE]
            + [TEXT_TOKEN_TYPE] * 3,
        ]
    )
    assert torch.equal(type_ids, expected)


def test_unified_sequence_has_expected_shapes_and_metadata() -> None:
    model = make_model()
    images = torch.randn(2, 1, 4, 4)
    token_ids = torch.tensor([[4, 5, 0], [7, 8, 9]])
    text_mask = make_text_padding_mask(token_ids, pad_token_id=0)

    output = model(images, token_ids, text_mask)

    assert output.embeddings.shape == (2, 8, 8)
    assert output.attention_mask.shape == (2, 8)
    assert output.token_type_ids.shape == (2, 8)
    assert output.visual_token_count == 4
    assert output.attention_mask[:, :5].all()
    assert torch.equal(output.attention_mask[:, 5:], text_mask)


def test_sequence_order_is_visual_then_separator_then_text() -> None:
    torch.manual_seed(13)
    model = make_model(embed_dim=4)
    images = torch.randn(1, 1, 4, 4)
    token_ids = torch.tensor([[3, 6]])
    text_mask = torch.ones_like(token_ids, dtype=torch.bool)
    with torch.no_grad():
        model.token_type_embedding.weight.zero_()
        model.sequence_position_embedding.weight.zero_()
        model.separator_token.fill_(7.0)

    expected_visual = model.vision_encoder(images)
    expected_text = model.text_embedding(token_ids)
    output = model(images, token_ids, text_mask).embeddings

    assert torch.allclose(output[:, :4], expected_visual)
    assert torch.equal(output[:, 4:5], torch.full((1, 1, 4), 7.0))
    assert torch.allclose(output[:, 5:], expected_text)


def test_type_and_position_embeddings_are_added_to_content() -> None:
    model = make_model(embed_dim=4)
    images = torch.randn(1, 1, 4, 4)
    token_ids = torch.tensor([[2]])
    text_mask = torch.ones_like(token_ids, dtype=torch.bool)
    with torch.no_grad():
        model.token_type_embedding.weight.zero_()
        model.sequence_position_embedding.weight.zero_()
    baseline = model(images, token_ids, text_mask).embeddings
    with torch.no_grad():
        model.token_type_embedding.weight[TEXT_TOKEN_TYPE].fill_(2.0)
        model.sequence_position_embedding.weight[0].fill_(3.0)

    changed = model(images, token_ids, text_mask).embeddings

    assert torch.allclose(changed[:, 0] - baseline[:, 0], torch.full((1, 4), 3.0))
    assert torch.allclose(changed[:, -1] - baseline[:, -1], torch.full((1, 4), 2.0))


def test_padding_slots_remain_in_sequence_but_are_masked() -> None:
    model = make_model()
    images = torch.randn(1, 1, 4, 4)
    token_ids = torch.tensor([[4, 0, 0]])
    text_mask = make_text_padding_mask(token_ids, pad_token_id=0)

    output = model(images, token_ids, text_mask)

    assert output.embeddings.shape[1] == 8
    assert torch.equal(
        output.attention_mask,
        torch.tensor([[True, True, True, True, True, True, False, False]]),
    )


@pytest.mark.parametrize(
    ("token_ids", "text_mask", "error"),
    [
        (torch.tensor([1, 2]), torch.tensor([True, True]), "2D"),
        (torch.ones(1, 6, dtype=torch.long), torch.ones(1, 6, dtype=torch.bool), "max"),
        (
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.bool),
            "shape",
        ),
        (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2), "boolean"),
    ],
)
def test_sequence_builder_rejects_invalid_text_inputs(
    token_ids: torch.Tensor,
    text_mask: torch.Tensor,
    error: str,
) -> None:
    model = make_model()

    with pytest.raises(ValueError, match=error):
        model(torch.randn(1, 1, 4, 4), token_ids, text_mask)


def test_sequence_builder_rejects_mismatched_batch_sizes() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="batch sizes"):
        model(
            torch.randn(2, 1, 4, 4),
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
        )


def test_constructor_requires_matching_embedding_widths() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        MultimodalSequenceEmbedding(
            vision_encoder=make_vision_encoder(embed_dim=8),
            vocab_size=16,
            max_text_tokens=4,
            embed_dim=6,
        )


def test_gradients_reach_visual_text_separator_type_and_position_parameters() -> None:
    model = make_model()
    images = torch.randn(2, 1, 4, 4)
    token_ids = torch.tensor([[2, 3], [4, 5]])
    text_mask = torch.ones_like(token_ids, dtype=torch.bool)

    model(images, token_ids, text_mask).embeddings.square().mean().backward()

    assert model.vision_encoder.patch_embedding.projection.weight.grad is not None
    assert model.text_embedding.weight.grad is not None
    assert model.separator_token.grad is not None
    assert model.token_type_embedding.weight.grad is not None
    assert model.sequence_position_embedding.weight.grad is not None
    assert model.text_embedding.weight.grad.abs().sum() > 0
    assert model.separator_token.grad.abs().sum() > 0
