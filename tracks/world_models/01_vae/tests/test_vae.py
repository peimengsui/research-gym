import torch

from implementation import VAE, vae_loss


def test_model_instantiates() -> None:
    model = VAE(input_dim=12, hidden_dim=8, latent_dim=3)

    assert isinstance(model, VAE)


def test_encode_returns_expected_shapes() -> None:
    model = VAE(input_dim=12, hidden_dim=8, latent_dim=3)
    x = torch.rand(4, 12)

    mu, logvar = model.encode(x)

    assert mu.shape == (4, 3)
    assert logvar.shape == (4, 3)


def test_reparameterize_returns_expected_shape() -> None:
    model = VAE(input_dim=12, hidden_dim=8, latent_dim=3)
    mu = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)

    z = model.reparameterize(mu, logvar)

    assert z.shape == (4, 3)
    assert torch.isfinite(z).all()


def test_decode_returns_valid_reconstruction() -> None:
    model = VAE(input_dim=12, hidden_dim=8, latent_dim=3)
    z = torch.randn(4, 3)

    reconstruction = model.decode(z)

    assert reconstruction.shape == (4, 12)
    assert torch.all((0 <= reconstruction) & (reconstruction <= 1))


def test_forward_returns_reconstruction_mu_and_logvar() -> None:
    model = VAE(input_dim=12, hidden_dim=8, latent_dim=3)
    x = torch.rand(4, 12)

    reconstruction, mu, logvar = model(x)

    assert reconstruction.shape == (4, 12)
    assert mu.shape == (4, 3)
    assert logvar.shape == (4, 3)


def test_vae_loss_returns_scalar_terms() -> None:
    reconstruction = torch.full((4, 12), 0.5)
    x = torch.rand(4, 12)
    mu = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)

    total, reconstruction_loss, kl_loss = vae_loss(
        reconstruction,
        x,
        mu,
        logvar,
    )

    assert total.shape == ()
    assert reconstruction_loss.shape == ()
    assert kl_loss.shape == ()
    assert torch.allclose(total, reconstruction_loss + kl_loss)
    assert kl_loss >= 0


def test_gradients_flow_through_model() -> None:
    torch.manual_seed(0)
    model = VAE(input_dim=12, hidden_dim=8, latent_dim=3)
    x = torch.rand(4, 12)

    reconstruction, mu, logvar = model(x)
    total, _, _ = vae_loss(reconstruction, x, mu, logvar)
    total.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0
