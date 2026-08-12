import pytest
import torch
import torch.nn.functional as F
from torch import nn

from implementation import (
    IGNORE_INDEX,
    MultimodalSelfAttention,
    MultimodalTransformerBlock,
    TinyNativeVLM,
    make_next_token_targets,
)


def make_model(num_multimodal_layers: int = 2) -> TinyNativeVLM:
    return TinyNativeVLM(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        vocab_size=12,
        max_text_tokens=4,
        embed_dim=8,
        num_heads=2,
        num_vision_layers=1,
        num_multimodal_layers=num_multimodal_layers,
    )


def test_multimodal_attention_uses_per_example_allow_matrix() -> None:
    attention = MultimodalSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, 6, dtype=torch.bool)
    mask[0, :, 5] = False
    mask[0, 5, :] = False

    output, weights = attention(x, mask, return_weights=True)

    assert output.shape == x.shape
    assert weights.shape == (2, 2, 6, 6)
    assert not weights[0, :, :, 5].any()
    assert not weights[0, :, 5, :].any()
    assert torch.allclose(weights[1].sum(dim=-1), torch.ones(2, 6))


def test_multimodal_attention_rejects_wrong_mask_shape() -> None:
    attention = MultimodalSelfAttention(embed_dim=8, num_heads=2)

    with pytest.raises(ValueError, match="attention_mask"):
        attention(torch.randn(2, 5, 8), torch.ones(5, 5, dtype=torch.bool))


def test_multimodal_block_preserves_residual_when_sublayers_are_zero() -> None:
    block = MultimodalTransformerBlock(embed_dim=8, num_heads=2)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
    x = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, 5, dtype=torch.bool)

    output = block(x, mask)

    assert torch.equal(output, x)


def test_next_token_targets_shift_and_ignore_padding() -> None:
    token_ids = torch.tensor([[1, 4, 2, 0], [1, 7, 8, 2]])
    validity = torch.tensor([[True, True, True, False], [True, True, True, True]])

    targets = make_next_token_targets(token_ids, validity)

    expected = torch.tensor(
        [[4, 2, IGNORE_INDEX, IGNORE_INDEX], [7, 8, 2, IGNORE_INDEX]]
    )
    assert torch.equal(targets, expected)


def test_next_token_targets_support_single_token_text() -> None:
    targets = make_next_token_targets(
        torch.tensor([[1], [2]]), torch.ones(2, 1, dtype=torch.bool)
    )

    assert torch.equal(targets, torch.full((2, 1), IGNORE_INDEX))


@pytest.mark.parametrize(
    ("token_ids", "validity"),
    [
        (torch.tensor([1, 2]), torch.tensor([True, True])),
        (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool)),
        (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2)),
    ],
)
def test_next_token_targets_reject_invalid_inputs(
    token_ids: torch.Tensor, validity: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        make_next_token_targets(token_ids, validity)


def test_tiny_vlm_returns_text_only_logits_and_layer_maps() -> None:
    model = make_model(num_multimodal_layers=2)
    images = torch.randn(3, 1, 4, 4)
    token_ids = torch.tensor([[1, 4, 2, 0], [1, 5, 2, 0], [1, 6, 7, 2]])
    validity = token_ids != 0

    output = model(images, token_ids, validity)

    assert output.logits.shape == (3, 4, 12)
    assert output.loss is None
    assert len(output.attention_maps) == 2
    assert all(weights.shape == (3, 2, 9, 9) for weights in output.attention_maps)
    assert not output.attention_maps[0][0, :, -1].any()


def test_future_text_does_not_change_earlier_text_logits() -> None:
    torch.manual_seed(15)
    model = make_model(num_multimodal_layers=1)
    image = torch.randn(1, 1, 4, 4)
    first = torch.tensor([[1, 4, 5, 2]])
    changed_future = torch.tensor([[1, 4, 9, 8]])
    validity = torch.ones(1, 4, dtype=torch.bool)

    first_logits = model(image, first, validity).logits
    changed_logits = model(image, changed_future, validity).logits

    assert torch.allclose(first_logits[:, :2], changed_logits[:, :2], atol=1e-6)


def test_loss_matches_cross_entropy_over_shifted_valid_targets() -> None:
    model = make_model(num_multimodal_layers=1)
    images = torch.randn(2, 1, 4, 4)
    token_ids = torch.tensor([[1, 4, 2, 0], [1, 7, 8, 2]])
    validity = token_ids != 0
    targets = make_next_token_targets(token_ids, validity)

    output = model(images, token_ids, validity, targets)
    expected = F.cross_entropy(
        output.logits.reshape(-1, 12),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )

    assert output.loss is not None
    assert torch.allclose(output.loss, expected)


def test_gradients_reach_vision_text_multimodal_and_output_paths() -> None:
    model = make_model(num_multimodal_layers=1)
    images = torch.randn(2, 1, 4, 4)
    token_ids = torch.tensor([[1, 4, 2, 0], [1, 7, 8, 2]])
    validity = token_ids != 0
    targets = make_next_token_targets(token_ids, validity)

    loss = model(images, token_ids, validity, targets).loss
    assert loss is not None
    loss.backward()

    assert (
        model.sequence_embedding.vision_encoder.patch_embedding.projection.weight.grad
        is not None
    )
    assert model.sequence_embedding.text_embedding.weight.grad is not None
    assert model.blocks[0].attention.query.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_tiny_vlm_rejects_invalid_targets() -> None:
    model = make_model()
    images = torch.randn(1, 1, 4, 4)
    token_ids = torch.tensor([[1, 4, 2, 0]])
    validity = token_ids != 0

    with pytest.raises(ValueError, match="targets"):
        model(images, token_ids, validity, torch.ones(1, 3, dtype=torch.long))
