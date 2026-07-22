import torch
from torch import nn

from implementation import (
    ConditionalTinyUNet,
    classifier_free_guidance,
    conditional_noise_prediction_loss,
    drop_class_conditions,
    gather_by_timestep,
    guided_ddim_sample_loop,
    make_diffusion_schedule,
)


def test_condition_dropout_keeps_or_drops_all_labels_at_endpoints() -> None:
    labels = torch.tensor([0, 1, 2, 1])

    kept, kept_mask = drop_class_conditions(labels, 0.0, null_class=3)
    dropped, dropped_mask = drop_class_conditions(labels, 1.0, null_class=3)

    assert torch.equal(kept, labels)
    assert not kept_mask.any()
    assert torch.equal(dropped, torch.full_like(labels, 3))
    assert dropped_mask.all()


def test_cfg_has_expected_endpoints_and_extrapolation() -> None:
    unconditional = torch.full((2, 1, 4, 4), 2.0)
    conditional = torch.full((2, 1, 4, 4), 5.0)

    assert torch.equal(
        classifier_free_guidance(unconditional, conditional, 0.0), unconditional
    )
    assert torch.equal(
        classifier_free_guidance(unconditional, conditional, 1.0), conditional
    )
    assert torch.equal(
        classifier_free_guidance(unconditional, conditional, 2.0),
        torch.full_like(unconditional, 8.0),
    )


def test_conditional_unet_output_shape_and_gradient_flow() -> None:
    torch.manual_seed(6)
    model = ConditionalTinyUNet(num_classes=3, base_channels=4, condition_dim=8)
    x_t = torch.randn(4, 1, 8, 8)
    timesteps = torch.tensor([0, 2, 4, 6])
    labels = torch.tensor([0, 1, 2, model.null_class])

    prediction = model(x_t, timesteps, labels)
    prediction.square().mean().backward()

    assert prediction.shape == x_t.shape
    assert model.class_embedding.weight.grad is not None
    assert model.input_conv.weight.grad is not None


def test_training_loss_is_scalar_and_backpropagates() -> None:
    torch.manual_seed(6)
    model = ConditionalTinyUNet(num_classes=2, base_channels=4, condition_dim=8)
    schedule = make_diffusion_schedule(8)
    images = torch.randn(4, 1, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])

    loss = conditional_noise_prediction_loss(
        model, schedule, images, labels, condition_drop_probability=0.5
    )
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert model.class_embedding.weight.grad is not None


class ConditionalOracle(nn.Module):
    """Return exact epsilon for either a class target or an all-zero null target."""

    def __init__(self, schedule, targets: torch.Tensor):
        super().__init__()
        self.schedule = schedule
        self.null_class = targets.shape[0]
        self.register_buffer("targets", targets)

    def forward(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        target = torch.zeros_like(x_t)
        real = labels != self.null_class
        target[real] = self.targets[labels[real]]
        signal = gather_by_timestep(
            self.schedule.sqrt_alpha_bars.to(x_t.device), timesteps, x_t.shape
        )
        noise_scale = gather_by_timestep(
            self.schedule.sqrt_one_minus_alpha_bars.to(x_t.device),
            timesteps,
            x_t.shape,
        )
        return (x_t - signal * target) / noise_scale


def test_guided_ddim_sampling_uses_requested_condition_and_scale() -> None:
    schedule = make_diffusion_schedule(10, beta_start=0.01, beta_end=0.1)
    targets = torch.zeros(2, 1, 8, 8)
    targets[0, :, :, 3:5] = 1.0
    targets[1, :, 3:5, :] = 1.0
    oracle = ConditionalOracle(schedule, targets)
    labels = torch.tensor([0, 1])
    initial_noise = torch.randn(2, 1, 8, 8)

    conditional_samples = guided_ddim_sample_loop(
        oracle,
        schedule,
        shape=(2, 1, 8, 8),
        class_labels=labels,
        guidance_scale=1.0,
        num_steps=5,
        initial_noise=initial_noise,
    )
    unconditional_samples = guided_ddim_sample_loop(
        oracle,
        schedule,
        shape=(2, 1, 8, 8),
        class_labels=labels,
        guidance_scale=0.0,
        num_steps=5,
        initial_noise=initial_noise,
    )

    assert torch.allclose(conditional_samples, targets, atol=1e-5)
    assert torch.allclose(unconditional_samples, torch.zeros_like(targets), atol=1e-5)
