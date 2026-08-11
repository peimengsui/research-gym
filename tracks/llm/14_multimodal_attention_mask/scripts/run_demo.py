"""Print the visual-prefix / causal-text attention allow-matrix."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        apply_attention_mask,
        make_multimodal_attention_matrix,
        prefix_length,
        visual_prefix_causal_mask,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        apply_attention_mask,
        make_multimodal_attention_matrix,
        prefix_length,
        visual_prefix_causal_mask,
    )


def _label_row(visual_token_count: int, text_tokens: int) -> list[str]:
    labels = [f"V{i}" for i in range(visual_token_count)]
    labels.append("S")
    labels.extend(f"T{i}" for i in range(text_tokens))
    return labels


def main() -> None:
    visual_token_count = 2
    text_tokens = 3
    seq = prefix_length(visual_token_count) + text_tokens
    labels = _label_row(visual_token_count, text_tokens)

    base = visual_prefix_causal_mask(seq, visual_token_count)
    # Example 0 has two padded text slots; example 1 is fully valid.
    validity = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )
    matrix = make_multimodal_attention_matrix(validity, visual_token_count)
    weights = apply_attention_mask(torch.zeros(2, seq, seq), matrix)

    print(f"sequence labels: {labels}")
    print(f"prefix length:   {prefix_length(visual_token_count)}")
    print("\nshared pattern (no padding):")
    print(base.to(dtype=torch.int64))
    print("\nbatched allow-matrix after padding:")
    print("example 0 validity:", validity[0].tolist())
    print(matrix[0].to(dtype=torch.int64))
    print("example 1 validity:", validity[1].tolist())
    print(matrix[1].to(dtype=torch.int64))
    print("\nuniform weights for example 0 (disallowed -> 0):")
    print(weights[0].round(decimals=3))
    print("Visual prefix is bidirectional; text is causal; padding is blocked.")


if __name__ == "__main__":
    main()
