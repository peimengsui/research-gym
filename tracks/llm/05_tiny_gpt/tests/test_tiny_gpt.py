import pytest
import torch
import torch.nn.functional as F
from torch import nn

from implementation import TinyGPT, TransformerBlock, make_lm_batch


def test_make_lm_batch_returns_next_token_targets() -> None:
    data = torch.arange(20)
    generator = torch.Generator().manual_seed(0)

    inputs, targets = make_lm_batch(
        data,
        block_size=5,
        batch_size=4,
        generator=generator,
    )

    assert inputs.shape == (4, 5)
    assert targets.shape == (4, 5)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert torch.all(targets[:, -1] == inputs[:, -1] + 1)


@pytest.mark.parametrize(
    ("data", "block_size", "batch_size"),
    [
        (torch.zeros(2, 3, dtype=torch.long), 2, 1),
        (torch.arange(4), 0, 1),
        (torch.arange(4), 2, 0),
        (torch.arange(4), 4, 1),
    ],
)
def test_make_lm_batch_rejects_invalid_inputs(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        make_lm_batch(data, block_size=block_size, batch_size=batch_size)


def test_tiny_gpt_has_expected_submodules() -> None:
    model = TinyGPT(vocab_size=11, block_size=6, embed_dim=8, num_layers=2)

    assert isinstance(model.token_embedding, nn.Embedding)
    assert isinstance(model.position_embedding, nn.Embedding)
    assert len(model.blocks) == 2
    assert all(isinstance(block, TransformerBlock) for block in model.blocks)
    assert isinstance(model.final_norm, nn.LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)


def test_tiny_gpt_rejects_invalid_constructor_arguments() -> None:
    with pytest.raises(ValueError):
        TinyGPT(vocab_size=0, block_size=4, embed_dim=8, num_layers=1)
    with pytest.raises(ValueError):
        TinyGPT(vocab_size=8, block_size=0, embed_dim=8, num_layers=1)
    with pytest.raises(ValueError):
        TinyGPT(vocab_size=8, block_size=4, embed_dim=0, num_layers=1)
    with pytest.raises(ValueError):
        TinyGPT(vocab_size=8, block_size=4, embed_dim=8, num_layers=0)


def test_forward_returns_logits_and_optional_loss() -> None:
    model = TinyGPT(vocab_size=13, block_size=5, embed_dim=8, num_layers=1)
    idx = torch.randint(0, 13, (3, 5))
    targets = torch.randint(0, 13, (3, 5))

    logits, loss = model(idx, targets)

    assert logits.shape == (3, 5, 13)
    assert loss is not None
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_forward_returns_no_loss_without_targets() -> None:
    model = TinyGPT(vocab_size=13, block_size=5, embed_dim=8, num_layers=1)
    idx = torch.randint(0, 13, (3, 5))

    logits, loss = model(idx)

    assert logits.shape == (3, 5, 13)
    assert loss is None


def test_forward_rejects_context_longer_than_block_size() -> None:
    model = TinyGPT(vocab_size=13, block_size=5, embed_dim=8, num_layers=1)

    with pytest.raises(ValueError):
        model(torch.randint(0, 13, (2, 6)))


def test_future_tokens_do_not_change_past_logits() -> None:
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=17, block_size=6, embed_dim=8, num_layers=2)
    idx = torch.randint(0, 17, (1, 6))
    changed_future = idx.clone()
    changed_future[:, -1] = (changed_future[:, -1] + 1) % 17

    original_logits, _ = model(idx)
    changed_logits, _ = model(changed_future)

    assert torch.allclose(original_logits[:, :-1, :], changed_logits[:, :-1, :])
    assert not torch.allclose(original_logits[:, -1, :], changed_logits[:, -1, :])


def test_generate_extends_sequence_and_preserves_prompt() -> None:
    torch.manual_seed(1)
    model = TinyGPT(vocab_size=9, block_size=4, embed_dim=8, num_layers=1)
    prompt = torch.tensor([[1, 2, 3]])

    generated = model.generate(prompt, max_new_tokens=5)

    assert generated.shape == (1, 8)
    assert torch.equal(generated[:, :3], prompt)
    assert torch.all((0 <= generated) & (generated < 9))


def test_generate_crops_context_to_block_size() -> None:
    torch.manual_seed(2)
    model = TinyGPT(vocab_size=9, block_size=3, embed_dim=8, num_layers=1)
    long_prompt = torch.tensor([[1, 2, 3, 4, 5]])

    generated = model.generate(long_prompt, max_new_tokens=2)

    assert generated.shape == (1, 7)
    assert torch.equal(generated[:, :5], long_prompt)


def test_gradients_flow_through_tiny_gpt() -> None:
    torch.manual_seed(3)
    model = TinyGPT(vocab_size=10, block_size=5, embed_dim=8, num_layers=1)
    idx = torch.randint(0, 10, (4, 5))
    targets = torch.randint(0, 10, (4, 5))

    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_model_can_fit_tiny_repeating_sequence() -> None:
    torch.manual_seed(4)
    model = TinyGPT(vocab_size=4, block_size=6, embed_dim=16, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    data = torch.tensor(([0, 1, 2, 3] * 24), dtype=torch.long)
    generator = torch.Generator().manual_seed(5)
    inputs, targets = make_lm_batch(
        data, block_size=6, batch_size=24, generator=generator
    )

    with torch.no_grad():
        _, initial_loss = model(inputs, targets)
        assert initial_loss is not None

    for _ in range(120):
        logits, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_logits, final_loss = model(inputs, targets)
        assert final_loss is not None

    assert final_loss < initial_loss * 0.2
    assert (
        F.cross_entropy(
            final_logits.reshape(-1, 4),
            targets.reshape(-1),
        )
        < initial_loss
    )
