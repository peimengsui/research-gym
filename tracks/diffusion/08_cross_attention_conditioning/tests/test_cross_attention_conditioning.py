import torch
from torch import nn

from implementation import (
    CrossAttentionLatentDenoiser,
    MultiHeadCrossAttention,
    TinyAutoencoder,
    TokenContextEncoder,
    context_conditioned_noise_loss,
    gather_by_timestep,
    make_diffusion_schedule,
    sample_conditioned_images,
)


def test_context_encoder_adds_token_and_position_embeddings() -> None:
    torch.manual_seed(8)
    encoder = TokenContextEncoder(vocab_size=10, max_length=4, context_dim=8)
    token_ids = torch.tensor([[3, 3, 3], [1, 2, 0]])

    context = encoder(token_ids)

    assert context.shape == (2, 3, 8)
    assert not torch.allclose(context[0, 0], context[0, 1])


def test_cross_attention_shape_gradient_and_padding_mask() -> None:
    torch.manual_seed(8)
    attention = MultiHeadCrossAttention(query_dim=8, context_dim=6, num_heads=2)
    queries = torch.randn(2, 4, 8, requires_grad=True)
    context = torch.randn(2, 3, 6, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    changed_context = context.detach().clone()
    changed_context[~mask] += 1_000.0

    output = attention(queries, context, mask)
    changed_output = attention(queries.detach(), changed_context, mask)
    output.square().mean().backward()

    assert output.shape == queries.shape
    assert torch.allclose(output.detach(), changed_output, atol=1e-5)
    assert queries.grad is not None
    assert context.grad is not None


def test_denoiser_output_depends_on_context() -> None:
    torch.manual_seed(8)
    denoiser = CrossAttentionLatentDenoiser(
        latent_channels=2,
        hidden_channels=8,
        time_embed_dim=8,
        context_dim=8,
        num_heads=2,
    )
    z_t = torch.randn(2, 2, 4, 4)
    timesteps = torch.tensor([2, 5])
    context_a = torch.zeros(2, 3, 8)
    context_b = torch.ones(2, 3, 8)

    output_a = denoiser(z_t, timesteps, context_a)
    output_b = denoiser(z_t, timesteps, context_b)

    assert output_a.shape == z_t.shape
    assert not torch.allclose(output_a, output_b)


def test_conditioned_loss_trains_context_and_denoiser_but_not_autoencoder() -> None:
    torch.manual_seed(8)
    autoencoder = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    context_encoder = TokenContextEncoder(12, 4, 8)
    denoiser = CrossAttentionLatentDenoiser(
        latent_channels=2,
        hidden_channels=8,
        time_embed_dim=8,
        context_dim=8,
        num_heads=2,
    )
    schedule = make_diffusion_schedule(8)
    images = torch.randn(3, 1, 8, 8)
    token_ids = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
    mask = token_ids != 0

    loss = context_conditioned_noise_loss(
        denoiser,
        context_encoder,
        autoencoder,
        schedule,
        images,
        token_ids,
        mask,
        latent_scale=2.0,
    )
    loss.backward()

    assert loss.shape == ()
    assert all(parameter.grad is None for parameter in autoencoder.parameters())
    assert all(parameter.grad is not None for parameter in context_encoder.parameters())
    assert all(parameter.grad is not None for parameter in denoiser.parameters())


class PromptOracleDenoiser(nn.Module):
    """Choose one clean latent target from the first context vector."""

    def __init__(self, schedule, target_a: torch.Tensor, target_b: torch.Tensor):
        super().__init__()
        self.schedule = schedule
        self.register_buffer("target_a", target_a)
        self.register_buffer("target_b", target_b)

    def forward(self, z_t, timesteps, context, context_mask=None):
        gate = context[:, 0, 0].reshape(-1, 1, 1, 1)
        clean = (1.0 - gate) * self.target_a + gate * self.target_b
        signal = gather_by_timestep(
            self.schedule.sqrt_alpha_bars.to(z_t.device), timesteps, z_t.shape
        )
        noise_scale = gather_by_timestep(
            self.schedule.sqrt_one_minus_alpha_bars.to(z_t.device),
            timesteps,
            z_t.shape,
        )
        return (z_t - signal * clean) / noise_scale


def test_same_noise_with_different_prompts_produces_different_latents() -> None:
    torch.manual_seed(8)
    autoencoder = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    context_encoder = TokenContextEncoder(4, 2, 4)
    with torch.no_grad():
        context_encoder.token_embedding.weight.zero_()
        context_encoder.position_embedding.weight.zero_()
        context_encoder.token_embedding.weight[2, 0] = 1.0
    target_a = torch.zeros(1, 2, 4, 4)
    target_a[:, :, :, 1:3] = 1.0
    target_b = torch.zeros(1, 2, 4, 4)
    target_b[:, :, 1:3, :] = 1.0
    schedule = make_diffusion_schedule(10, beta_start=0.01, beta_end=0.1)
    oracle = PromptOracleDenoiser(schedule, target_a, target_b)
    token_ids = torch.tensor([[1], [2]])
    initial_noise = torch.randn(1, 2, 4, 4).expand(2, -1, -1, -1).clone()

    images, latents = sample_conditioned_images(
        autoencoder,
        context_encoder,
        oracle,
        schedule,
        image_shape=(2, 1, 8, 8),
        token_ids=token_ids,
        context_mask=torch.ones_like(token_ids, dtype=torch.bool),
        latent_scale=1.0,
        num_steps=5,
        initial_noise=initial_noise,
    )

    assert images.shape == (2, 1, 8, 8)
    assert torch.allclose(latents[0], target_a[0], atol=1e-5)
    assert torch.allclose(latents[1], target_b[0], atol=1e-5)
