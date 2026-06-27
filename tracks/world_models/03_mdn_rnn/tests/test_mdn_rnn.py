import math

import pytest
import torch

from implementation import (
    MDNRNN,
    gaussian_mixture_nll,
    make_sequence_batch,
    mixture_mean,
)


def test_make_sequence_batch_slices_aligned_inputs() -> None:
    latents = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            [[10.0, 11.0], [11.0, 12.0], [12.0, 13.0]],
        ]
    )
    actions = torch.tensor(
        [
            [[0.5], [1.5]],
            [[2.5], [3.5]],
        ]
    )

    current_z, current_actions, next_z = make_sequence_batch(latents, actions)

    assert current_z.shape == (2, 2, 2)
    assert current_actions.shape == (2, 2, 1)
    assert next_z.shape == (2, 2, 2)
    assert torch.equal(current_z[0, 0], torch.tensor([0.0, 1.0]))
    assert torch.equal(current_actions[1, 0], torch.tensor([2.5]))
    assert torch.equal(next_z[-1, -1], torch.tensor([12.0, 13.0]))


@pytest.mark.parametrize(
    ("latents", "actions"),
    [
        (torch.zeros(2, 3), torch.zeros(2, 2, 1)),
        (torch.zeros(2, 3, 2), torch.zeros(2, 2)),
        (torch.zeros(2, 3, 2), torch.zeros(3, 2, 1)),
        (torch.zeros(2, 3, 2), torch.zeros(2, 3, 1)),
    ],
)
def test_make_sequence_batch_rejects_misaligned_inputs(
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        make_sequence_batch(latents, actions)


def test_model_rejects_non_positive_mixture_count() -> None:
    with pytest.raises(ValueError):
        MDNRNN(latent_dim=2, action_dim=1, hidden_dim=8, num_mixtures=0)


def test_forward_returns_expected_mixture_shapes() -> None:
    model = MDNRNN(latent_dim=3, action_dim=2, hidden_dim=7, num_mixtures=4)
    latents = torch.randn(5, 6, 3)
    actions = torch.randn(5, 6, 2)

    logits, mu, log_sigma, next_hidden = model(latents, actions)

    assert logits.shape == (5, 6, 4)
    assert mu.shape == (5, 6, 4, 3)
    assert log_sigma.shape == (5, 6, 4, 3)
    assert next_hidden.shape == (1, 5, 7)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_sigma).all()


def test_forward_accepts_previous_hidden_state() -> None:
    model = MDNRNN(latent_dim=2, action_dim=1, hidden_dim=5, num_mixtures=3)
    latents = torch.randn(4, 3, 2)
    actions = torch.randn(4, 3, 1)
    hidden = torch.randn(1, 4, 5)

    _, _, _, next_hidden = model(latents, actions, hidden)

    assert next_hidden.shape == hidden.shape


def test_mixture_mean_combines_component_means_with_probabilities() -> None:
    logits = torch.tensor([[[2.0, 0.0]]])
    mu = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])

    prediction = mixture_mean(logits, mu)

    probabilities = torch.softmax(logits, dim=-1)
    expected = (
        probabilities[..., :1] * mu[:, :, 0, :]
        + probabilities[..., 1:] * mu[:, :, 1, :]
    )
    assert prediction.shape == (1, 1, 2)
    assert torch.allclose(prediction, expected)


def test_gaussian_mixture_nll_matches_single_component_gaussian() -> None:
    logits = torch.zeros(1, 1, 1)
    mu = torch.zeros(1, 1, 1, 2)
    log_sigma = torch.zeros(1, 1, 1, 2)
    target = torch.zeros(1, 1, 2)

    loss = gaussian_mixture_nll(logits, mu, log_sigma, target)

    assert loss.shape == ()
    assert torch.allclose(loss, torch.tensor(math.log(2 * math.pi)))


def test_gaussian_mixture_nll_prefers_closer_component() -> None:
    logits = torch.zeros(1, 1, 2)
    mu = torch.tensor([[[[0.0], [10.0]]]])
    log_sigma = torch.zeros(1, 1, 2, 1)

    near_loss = gaussian_mixture_nll(logits, mu, log_sigma, torch.tensor([[[0.0]]]))
    far_loss = gaussian_mixture_nll(logits, mu, log_sigma, torch.tensor([[[5.0]]]))

    assert near_loss < far_loss


def test_recurrent_model_is_causal_over_time() -> None:
    torch.manual_seed(0)
    model = MDNRNN(latent_dim=2, action_dim=1, hidden_dim=6, num_mixtures=2)
    latents = torch.randn(1, 5, 2)
    actions = torch.randn(1, 5, 1)
    changed_future = latents.clone()
    changed_future[:, -1, :] += 100.0

    original_logits, original_mu, _, _ = model(latents, actions)
    changed_logits, changed_mu, _, _ = model(changed_future, actions)

    assert torch.allclose(original_logits[:, :-1, :], changed_logits[:, :-1, :])
    assert torch.allclose(original_mu[:, :-1, :, :], changed_mu[:, :-1, :, :])
    assert not torch.allclose(original_mu[:, -1, :, :], changed_mu[:, -1, :, :])


def test_gradients_flow_through_mdn_loss() -> None:
    torch.manual_seed(1)
    model = MDNRNN(latent_dim=2, action_dim=1, hidden_dim=8, num_mixtures=3)
    latents = torch.randn(4, 5, 2)
    actions = torch.randn(4, 5, 1)
    target = torch.randn(4, 5, 2)

    logits, mu, log_sigma, _ = model(latents, actions)
    loss = gaussian_mixture_nll(logits, mu, log_sigma, target)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_model_can_fit_simple_sequence_dynamics() -> None:
    torch.manual_seed(2)
    model = MDNRNN(latent_dim=1, action_dim=1, hidden_dim=12, num_mixtures=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    latents = torch.zeros(16, 7, 1)
    actions = torch.randn(16, 6, 1)
    for step in range(actions.shape[1]):
        latents[:, step + 1, :] = 0.8 * latents[:, step, :] + 0.5 * actions[:, step, :]
    current_z, current_actions, next_z = make_sequence_batch(latents, actions)

    with torch.no_grad():
        logits, mu, log_sigma, _ = model(current_z, current_actions)
        initial_loss = gaussian_mixture_nll(logits, mu, log_sigma, next_z)

    for _ in range(140):
        logits, mu, log_sigma, _ = model(current_z, current_actions)
        loss = gaussian_mixture_nll(logits, mu, log_sigma, next_z)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, mu, log_sigma, _ = model(current_z, current_actions)
        final_loss = gaussian_mixture_nll(logits, mu, log_sigma, next_z)

    assert final_loss < initial_loss
