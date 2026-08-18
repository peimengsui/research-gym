import pytest
import torch
import torch.nn.functional as F

from implementation import (
    CachedMultimodalSelfAttention,
    TinyCachedVLM,
    generate_multimodal,
    make_visual_prefix_attention_mask,
)


def make_model(max_text_tokens: int = 8, num_layers: int = 2) -> TinyCachedVLM:
    return TinyCachedVLM(
        image_size=4,
        patch_size=2,
        in_channels=1,
        vocab_size=11,
        max_text_tokens=max_text_tokens,
        embed_dim=8,
        num_heads=2,
        num_layers=num_layers,
    )


def test_visual_prefix_mask_is_dense_then_text_causal() -> None:
    mask = make_visual_prefix_attention_mask(
        batch_size=2, visual_token_count=2, text_token_count=3
    )
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

    assert mask.shape == (2, 6, 6)
    assert mask.dtype == torch.bool
    assert torch.equal(mask[0], expected)
    assert torch.equal(mask[1], expected)


@pytest.mark.parametrize(
    ("batch_size", "visual_tokens", "text_tokens"),
    [(0, 2, 2), (1, 0, 2), (1, 2, 0)],
)
def test_visual_prefix_mask_rejects_non_positive_lengths(
    batch_size: int, visual_tokens: int, text_tokens: int
) -> None:
    with pytest.raises(ValueError):
        make_visual_prefix_attention_mask(batch_size, visual_tokens, text_tokens)


def test_cached_attention_appends_projected_keys_and_values() -> None:
    attention = CachedMultimodalSelfAttention(embed_dim=8, num_heads=2)
    first = torch.randn(2, 3, 8)
    second = torch.randn(2, 1, 8)

    first_output, first_cache = attention(first, torch.ones(2, 3, 3, dtype=torch.bool))
    second_output, second_cache = attention(
        second, torch.ones(2, 1, 4, dtype=torch.bool), first_cache
    )

    assert first_output.shape == (2, 3, 8)
    assert second_output.shape == (2, 1, 8)
    assert first_cache[0].shape == (2, 2, 3, 4)
    assert first_cache[1].shape == (2, 2, 3, 4)
    assert second_cache[0].shape == (2, 2, 4, 4)
    assert torch.equal(second_cache[0][:, :, :3], first_cache[0])
    assert torch.equal(second_cache[1][:, :, :3], first_cache[1])


def test_prefill_returns_text_logits_and_one_cache_per_layer() -> None:
    model = make_model(num_layers=2)
    images = torch.randn(2, 1, 4, 4)
    prompt = torch.tensor([[1, 3, 4], [1, 5, 4]])

    logits, caches = model.prefill(images, prompt)

    # Four image patches + separator + three prompt tokens.
    assert logits.shape == (2, 3, 11)
    assert len(caches) == 2
    assert all(key.shape == (2, 2, 8, 4) for key, _ in caches)
    assert all(value.shape == key.shape for key, value in caches)


def test_decode_step_matches_full_recomputation_for_appended_token() -> None:
    torch.manual_seed(17)
    model = make_model()
    images = torch.randn(2, 1, 4, 4)
    prompt = torch.tensor([[1, 3, 4], [1, 5, 4]])
    appended = torch.tensor([[7], [8]])

    _, caches = model.prefill(images, prompt)
    cached_logits, new_caches = model.decode_step(appended, caches)
    full_logits = model.forward_full(images, torch.cat((prompt, appended), dim=1))

    assert cached_logits.shape == (2, 1, 11)
    assert torch.allclose(cached_logits[:, 0], full_logits[:, -1], atol=1e-5)
    assert all(cache[0].shape[2] == 9 for cache in new_caches)


def naive_greedy_generation(
    model: TinyCachedVLM,
    images: torch.Tensor,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    generated = prompt
    for _ in range(max_new_tokens):
        logits = model.forward_full(images, generated)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return generated


def test_cached_generation_matches_full_recomputation() -> None:
    torch.manual_seed(18)
    model = make_model(max_text_tokens=9)
    images = torch.randn(2, 1, 4, 4)
    prompt = torch.tensor([[1, 3, 4], [1, 5, 4]])

    cached = generate_multimodal(model, images, prompt, max_new_tokens=4)
    naive = naive_greedy_generation(model, images, prompt, max_new_tokens=4)

    assert torch.equal(cached, naive)


def test_cached_generation_embeds_images_only_once() -> None:
    model = make_model()
    images = torch.randn(2, 1, 4, 4)
    prompt = torch.tensor([[1, 3, 4], [1, 5, 4]])
    calls = 0

    def count_calls(
        module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        del module, inputs, output
        nonlocal calls
        calls += 1

    handle = model.vision_embedding.register_forward_hook(count_calls)
    generate_multimodal(model, images, prompt, max_new_tokens=3)
    handle.remove()

    assert calls == 1


def test_generation_stops_when_every_row_emits_eos() -> None:
    model = make_model()
    images = torch.randn(2, 1, 4, 4)
    prompt = torch.tensor([[1, 3, 4], [1, 5, 4]])
    eos_token_id = 2
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[eos_token_id] = 1.0

    generated = generate_multimodal(
        model, images, prompt, max_new_tokens=4, eos_token_id=eos_token_id
    )

    assert generated.shape == (2, 4)
    assert torch.equal(generated[:, -1], torch.full((2,), eos_token_id))


def test_generation_rejects_text_beyond_model_capacity() -> None:
    model = make_model(max_text_tokens=5)
    images = torch.randn(1, 1, 4, 4)
    prompt = torch.tensor([[1, 3, 4]])

    with pytest.raises(ValueError, match="max_text_tokens"):
        generate_multimodal(model, images, prompt, max_new_tokens=3)


def test_full_path_gradients_reach_vision_attention_and_head() -> None:
    torch.manual_seed(19)
    model = make_model(num_layers=1)
    images = torch.randn(2, 1, 4, 4)
    prompt = torch.tensor([[1, 3, 4], [1, 5, 4]])
    targets = torch.randint(0, model.vocab_size, prompt.shape)

    logits = model.forward_full(images, prompt)
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
    loss.backward()

    assert model.vision_embedding.projection.weight.grad is not None
    assert model.blocks[0].attention.key.weight.grad is not None
    assert model.text_embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.vision_embedding.projection.weight.grad.abs().sum() > 0
