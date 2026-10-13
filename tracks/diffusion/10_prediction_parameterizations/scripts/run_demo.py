"""Compare diffusion targets and timestep-dependent Min-SNR weights."""

import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    min_snr_loss_weights,
    prediction_target,
    prediction_to_x0_and_epsilon,
    snr_from_alpha_bar,
)
from provided import diffusion_coefficients  # noqa: E402


def main() -> None:
    torch.manual_seed(10)
    x_start = torch.randn(3, 1, 4, 4)
    noise = torch.randn_like(x_start)
    alpha_bar = torch.tensor([0.95, 0.5, 0.05])
    alpha, sigma = diffusion_coefficients(alpha_bar, x_start)
    x_t = alpha * x_start + sigma * noise

    print("alpha_bar       SNR       epsilon_w  sample_w  velocity_w")
    snr = snr_from_alpha_bar(alpha_bar)
    weights = {
        name: min_snr_loss_weights(alpha_bar, name, gamma=5.0)
        for name in ("epsilon", "sample", "velocity")
    }
    for index in range(alpha_bar.shape[0]):
        print(
            f"{alpha_bar[index]:8.3f}  {snr[index]:8.3f}  "
            f"{weights['epsilon'][index]:9.3f}  "
            f"{weights['sample'][index]:8.3f}  "
            f"{weights['velocity'][index]:10.3f}"
        )

    print("\nround-trip maximum errors:")
    for prediction_type in ("epsilon", "sample", "velocity"):
        target = prediction_target(x_start, noise, alpha_bar, prediction_type)
        recovered_x_start, recovered_noise = prediction_to_x0_and_epsilon(
            x_t,
            target,
            alpha_bar,
            prediction_type,
        )
        error = max(
            (recovered_x_start - x_start).abs().max().item(),
            (recovered_noise - noise).abs().max().item(),
        )
        print(f"  {prediction_type:8s}: {error:.7f}")


if __name__ == "__main__":
    main()
