"""Compare floating and simulated int8 weight-only token models."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        WeightOnlyQuantizedLinear,
        make_quantized_toy_model,
        module_storage_bytes,
    )
    from provided import TinyTokenModel
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        WeightOnlyQuantizedLinear,
        make_quantized_toy_model,
        module_storage_bytes,
    )
    from provided import TinyTokenModel


def main() -> None:
    torch.manual_seed(19)
    model = TinyTokenModel(
        vocab_size=64,
        embed_dim=24,
        hidden_dim=48,
        num_layers=2,
    )
    token_ids = torch.randint(0, 64, (4, 8))
    quantized_model = make_quantized_toy_model(model, per_channel=True)

    with torch.no_grad():
        float_logits = model(token_ids)
        quantized_logits = quantized_model(token_ids)

    float_bytes = module_storage_bytes(model)
    quantized_bytes = module_storage_bytes(quantized_model)
    absolute_error = (quantized_logits - float_logits).abs()
    float_predictions = float_logits.argmax(dim=-1)
    quantized_predictions = quantized_logits.argmax(dim=-1)
    quantized_layer_count = sum(
        isinstance(module, WeightOnlyQuantizedLinear)
        for module in quantized_model.modules()
    )

    print(f"quantized linear layers: {quantized_layer_count}")
    print(f"floating storage:        {float_bytes:,} bytes")
    print(f"quantized storage:       {quantized_bytes:,} bytes")
    print(f"storage ratio:           {quantized_bytes / float_bytes:.3f}")
    print(f"mean logit error:        {absolute_error.mean().item():.6f}")
    print(f"max logit error:         {absolute_error.max().item():.6f}")
    print(
        "argmax agreement:        "
        f"{(float_predictions == quantized_predictions).float().mean().item():.3f}"
    )
    print("Weights are stored as int8 but dequantized for ordinary F.linear compute.")


if __name__ == "__main__":
    main()
