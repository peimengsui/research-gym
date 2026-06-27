import torch
import torch.nn.functional as F
from torch import nn

from implementation import CausalSelfAttention, FeedForward, TransformerBlock


def test_feed_forward_uses_expanded_hidden_dimension() -> None:
    feed_forward = FeedForward(embed_dim=6, expansion_factor=3)

    first_linear = feed_forward.network[0]
    second_linear = feed_forward.network[2]
    assert isinstance(first_linear, nn.Linear)
    assert isinstance(feed_forward.network[1], nn.GELU)
    assert isinstance(second_linear, nn.Linear)
    assert first_linear.in_features == 6
    assert first_linear.out_features == 18
    assert second_linear.in_features == 18
    assert second_linear.out_features == 6


def test_feed_forward_preserves_sequence_shape() -> None:
    feed_forward = FeedForward(embed_dim=6, expansion_factor=4)
    x = torch.randn(2, 5, 6)

    output = feed_forward(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_transformer_block_has_expected_sublayers() -> None:
    block = TransformerBlock(embed_dim=8, expansion_factor=2)

    assert isinstance(block.attention, CausalSelfAttention)
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)
    assert isinstance(block.feed_forward, FeedForward)


def test_transformer_block_returns_output_and_attention_weights() -> None:
    block = TransformerBlock(embed_dim=8, expansion_factor=2)
    x = torch.randn(3, 4, 8)

    output, weights = block(x, return_weights=True)

    assert output.shape == x.shape
    assert weights.shape == (3, 4, 4)
    assert torch.isfinite(output).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 4))


def test_transformer_block_can_return_only_output() -> None:
    block = TransformerBlock(embed_dim=8, expansion_factor=2)
    x = torch.randn(3, 4, 8)

    output = block(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape


def test_future_tokens_do_not_change_past_outputs() -> None:
    torch.manual_seed(0)
    block = TransformerBlock(embed_dim=4, expansion_factor=2)
    x = torch.randn(1, 5, 4)
    changed_future = x.clone()
    changed_future[:, 4, :] += 100.0

    original = block(x)
    changed = block(changed_future)

    assert torch.allclose(original[:, :4, :], changed[:, :4, :], atol=1e-5)
    assert not torch.allclose(original[:, 4, :], changed[:, 4, :], atol=1e-5)


def test_zero_sublayers_leave_residual_stream_unchanged() -> None:
    block = TransformerBlock(embed_dim=4, expansion_factor=2)
    for parameter in block.parameters():
        parameter.data.zero_()
    x = torch.randn(2, 3, 4)

    output = block(x)

    assert torch.equal(output, x)


def test_gradients_flow_through_transformer_block() -> None:
    torch.manual_seed(1)
    block = TransformerBlock(embed_dim=4, expansion_factor=2)
    x = torch.randn(3, 5, 4)
    target = torch.zeros_like(x)

    output = block(x)
    loss = F.mse_loss(output, target)
    loss.backward()

    gradients = [
        parameter.grad for parameter in block.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_block_can_learn_simple_residual_correction() -> None:
    torch.manual_seed(2)
    block = TransformerBlock(embed_dim=3, expansion_factor=2)
    optimizer = torch.optim.Adam(block.parameters(), lr=0.03)
    x = torch.randn(16, 4, 3)
    target = x * 0.5

    with torch.no_grad():
        initial_loss = F.mse_loss(block(x), target)

    for _ in range(120):
        output = block(x)
        loss = F.mse_loss(output, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = F.mse_loss(block(x), target)

    assert final_loss < initial_loss * 0.2
