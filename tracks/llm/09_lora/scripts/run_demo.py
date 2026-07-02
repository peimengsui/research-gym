"""Train a tiny LoRA adapter while the base layer stays frozen."""

import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        LoRALinear,
        count_parameters,
        count_trainable_parameters,
        make_low_rank_regression_data,
        merge_lora_weights,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py",
            WORKSPACE_ROOT / "implementation.py",
        )
    from implementation import (
        LoRALinear,
        count_parameters,
        count_trainable_parameters,
        make_low_rank_regression_data,
        merge_lora_weights,
    )


def main() -> None:
    torch.manual_seed(9)
    x, target, base = make_low_rank_regression_data(
        n_samples=96,
        in_features=6,
        out_features=4,
        rank=2,
    )
    layer = LoRALinear.from_linear(base, rank=2, alpha=2.0)
    optimizer = torch.optim.Adam(
        [parameter for parameter in layer.parameters() if parameter.requires_grad],
        lr=0.08,
    )

    print(f"total parameters:     {count_parameters(layer)}")
    print(f"trainable parameters: {count_trainable_parameters(layer)}")
    print("base layer is frozen; only LoRA A and B are trainable.")

    with torch.no_grad():
        initial_loss = F.mse_loss(layer(x), target)
    print(f"initial MSE: {initial_loss.item():.4f}")

    for step in range(1, 181):
        loss = F.mse_loss(layer(x), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, 30, 90, 180}:
            print(f"step {step:03d} | MSE={loss.item():.4f}")

    merged = merge_lora_weights(layer)
    with torch.no_grad():
        final_loss = F.mse_loss(layer(x), target)
        merge_error = (merged(x) - layer(x)).abs().max()
    print(f"final MSE:       {final_loss.item():.4f}")
    print(f"merge max error: {merge_error.item():.8f}")
    print("LoRA learned a low-rank update without changing the base weight.")


if __name__ == "__main__":
    main()
