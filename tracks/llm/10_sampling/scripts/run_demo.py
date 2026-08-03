"""Compare sampling policies on one fixed next-token distribution."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import sample_next_token
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import sample_next_token


def sample_counts(
    logits: torch.Tensor,
    draws: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> list[int]:
    repeated_logits = logits.expand(draws, -1)
    tokens = sample_next_token(
        repeated_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=torch.Generator().manual_seed(10),
    )
    return torch.bincount(tokens[:, 0], minlength=logits.shape[-1]).tolist()


def main() -> None:
    logits = torch.tensor([[2.0, 1.2, 0.8, 0.2, -0.5]])
    draws = 1_000
    print("token IDs:                 [0, 1, 2, 3, 4]")
    print(f"temperature=0.5 counts:   {sample_counts(logits, draws, temperature=0.5)}")
    print(f"temperature=1.5 counts:   {sample_counts(logits, draws, temperature=1.5)}")
    print(f"top-k=2 counts:           {sample_counts(logits, draws, top_k=2)}")
    print(f"top-p=0.75 counts:        {sample_counts(logits, draws, top_p=0.75)}")
    greedy = sample_next_token(logits, do_sample=False).item()
    print(f"greedy token:             {greedy}")
    print("Lower temperature concentrates samples; top-k and top-p remove candidates.")


if __name__ == "__main__":
    main()
