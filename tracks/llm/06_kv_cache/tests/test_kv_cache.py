import pytest
import torch

from implementation import (
    CachedCausalSelfAttention,
    TinyCachedGPT,
    causal_mask,
)


def test_causal_mask_supports_cached_keys() -> None:
    mask = causal_mask(query_length=2, key_length=5)
    expected = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )

    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


@pytest.mark.parametrize(
    ("query_length", "key_length"),
    [(0, None), (3, 2)],
)
def test_causal_mask_rejects_invalid_lengths(
    query_length: int,
    key_length: int | None,
) -> None:
    with pytest.raises(ValueError):
        causal_mask(query_length, key_length)


def test_cached_attention_appends_key_value_cache() -> None:
    attention = CachedCausalSelfAttention(embed_dim=4)
    x1 = torch.randn(2, 3, 4)
    x2 = torch.randn(2, 1, 4)

    out1, cache1 = attention(x1)
    out2, cache2 = attention(x2, cache1)

    assert out1.shape == (2, 3, 4)
    assert out2.shape == (2, 1, 4)
    assert cache1[0].shape == (2, 3, 4)
    assert cache1[1].shape == (2, 3, 4)
    assert cache2[0].shape == (2, 4, 4)
    assert cache2[1].shape == (2, 4, 4)
    assert torch.equal(cache2[0][:, :3, :], cache1[0])
    assert torch.equal(cache2[1][:, :3, :], cache1[1])


def test_forward_and_step_logits_match_for_prompt() -> None:
    torch.manual_seed(0)
    model = TinyCachedGPT(vocab_size=13, block_size=8, embed_dim=8, num_layers=2)
    idx = torch.randint(0, 13, (2, 5))

    full_logits, _ = model(idx)
    caches = None
    step_logits = []
    for position in range(idx.shape[1]):
        logits, caches = model.forward_step(
            idx[:, position : position + 1],
            caches,
            position_offset=position,
        )
        step_logits.append(logits)

    cached_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(cached_logits, full_logits, atol=1e-5)
    assert caches is not None
    assert len(caches) == 2


def test_forward_step_rejects_misaligned_cache_count() -> None:
    model = TinyCachedGPT(vocab_size=13, block_size=8, embed_dim=8, num_layers=2)
    idx = torch.randint(0, 13, (2, 1))
    bad_cache = [(torch.zeros(2, 1, 8), torch.zeros(2, 1, 8))]

    with pytest.raises(ValueError):
        model.forward_step(idx, bad_cache, position_offset=1)


def test_forward_rejects_too_long_context() -> None:
    model = TinyCachedGPT(vocab_size=13, block_size=4, embed_dim=8, num_layers=1)

    with pytest.raises(ValueError):
        model(torch.randint(0, 13, (2, 5)))


def test_cached_generation_matches_full_context_greedy_generation() -> None:
    torch.manual_seed(1)
    model = TinyCachedGPT(vocab_size=11, block_size=8, embed_dim=8, num_layers=2)
    prompt = torch.tensor([[1, 2, 3]])

    full = model.generate_full(prompt, max_new_tokens=4)
    cached = model.generate_cached(prompt, max_new_tokens=4)

    assert torch.equal(cached, full)


def test_cached_generation_rejects_sequences_beyond_block_size() -> None:
    model = TinyCachedGPT(vocab_size=11, block_size=5, embed_dim=8, num_layers=1)
    prompt = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError):
        model.generate_cached(prompt, max_new_tokens=3)


def test_cached_generation_preserves_prompt_and_extends() -> None:
    torch.manual_seed(2)
    model = TinyCachedGPT(vocab_size=9, block_size=7, embed_dim=8, num_layers=1)
    prompt = torch.tensor([[1, 2, 3]])

    generated = model.generate_cached(prompt, max_new_tokens=2)

    assert generated.shape == (1, 5)
    assert torch.equal(generated[:, :3], prompt)
    assert torch.all((0 <= generated) & (generated < 9))


def test_gradients_flow_through_full_context_training_path() -> None:
    torch.manual_seed(3)
    model = TinyCachedGPT(vocab_size=10, block_size=5, embed_dim=8, num_layers=1)
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
