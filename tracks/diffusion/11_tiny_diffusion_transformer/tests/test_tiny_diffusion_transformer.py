import pytest
import torch

from implementation import (
    TinyDiTBlock,
    TinyDiffusionTransformer,
    diffusion_transformer_loss,
    patchify,
    unpatchify,
)
from provided import make_diffusion_schedule, make_tiny_latents


def test_patchify_uses_row_major_patch_order() -> None:
    samples = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    patches = patchify(samples, patch_size=2)

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 4.0, 5.0],
                [2.0, 3.0, 6.0, 7.0],
                [8.0, 9.0, 12.0, 13.0],
                [10.0, 11.0, 14.0, 15.0],
            ]
        ]
    )
    assert torch.equal(patches, expected)


def test_unpatchify_exactly_inverts_rectangular_multichannel_input() -> None:
    torch.manual_seed(11)
    samples = torch.randn(2, 3, 4, 6)

    patches = patchify(samples, patch_size=2)
    reconstructed = unpatchify(patches, 3, 4, 6, 2)

    assert patches.shape == (2, 6, 12)
    assert torch.equal(reconstructed, samples)


def test_modulation_broadcasts_scale_and_shift_over_patch_tokens() -> None:
    normalized_tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    shift = torch.tensor([[10.0, 20.0]])
    scale = torch.tensor([[1.0, 2.0]])

    modulated = TinyDiTBlock._modulate(normalized_tokens, shift, scale)

    expected = torch.tensor([[[12.0, 26.0], [16.0, 32.0]]])
    assert torch.equal(modulated, expected)


def test_adaptive_block_preserves_shape_and_uses_conditioning() -> None:
    torch.manual_seed(11)
    block = TinyDiTBlock(embedding_dim=16, num_heads=4)
    tokens = torch.randn(2, 5, 16)
    first_condition = torch.zeros(2, 16)
    second_condition = torch.ones(2, 16)

    first = block(tokens, first_condition)
    second = block(tokens, second_condition)

    assert first.shape == tokens.shape
    assert not torch.allclose(first, second)


def test_tiny_dit_returns_sample_shape_and_names_position_embedding() -> None:
    torch.manual_seed(11)
    model = TinyDiffusionTransformer(
        image_size=8,
        patch_size=2,
        channels=2,
        embedding_dim=16,
        depth=1,
        num_heads=4,
    )
    noisy_samples = torch.randn(3, 2, 8, 8)
    timesteps = torch.tensor([0, 4, 9])

    prediction = model(noisy_samples, timesteps)

    assert prediction.shape == noisy_samples.shape
    assert model.position_embedding.shape == (1, 16, 16)


def test_timestep_changes_the_denoising_prediction() -> None:
    torch.manual_seed(12)
    model = TinyDiffusionTransformer(embedding_dim=16, depth=1, num_heads=4)
    noisy_sample = torch.randn(1, 2, 8, 8).expand(2, -1, -1, -1).clone()

    prediction = model(noisy_sample, torch.tensor([0, 9]))

    assert not torch.allclose(prediction[0], prediction[1])


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "velocity"])
def test_denoising_loss_backpropagates_for_each_target(prediction_type: str) -> None:
    torch.manual_seed(13)
    model = TinyDiffusionTransformer(embedding_dim=16, depth=1, num_heads=4)
    schedule = make_diffusion_schedule(10)
    clean_samples = make_tiny_latents(n=3)
    timesteps = torch.tensor([1, 5, 9])
    noise = torch.randn_like(clean_samples)

    loss = diffusion_transformer_loss(
        model,
        schedule,
        clean_samples,
        prediction_type,
        timesteps,
        noise,
    )
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert model.patch_projection.weight.grad is not None
    assert model.blocks[0].attention.in_proj_weight.grad is not None
    assert model.time_mlp[0].weight.grad is not None
    assert model.output_projection.weight.grad is not None


def test_patchify_rejects_non_divisible_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        patchify(torch.randn(1, 2, 5, 4), patch_size=2)
