"""Compare target-only sampling with exact speculative decoding statistics."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import sample_token, speculative_generate
    from provided import make_toy_speculative_models
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import sample_token, speculative_generate
    from provided import make_toy_speculative_models


def target_only_generate(target, prompt, max_new_tokens, generator):
    generated = prompt.clone()
    for _ in range(max_new_tokens):
        token = sample_token(target.next_probabilities(generated), generator)
        generated = torch.cat((generated, torch.tensor([token])))
    return generated


def main() -> None:
    prompt = torch.tensor([0], dtype=torch.long)
    max_new_tokens = 12
    target, draft = make_toy_speculative_models()
    baseline = target_only_generate(
        target,
        prompt,
        max_new_tokens,
        torch.Generator().manual_seed(0),
    )
    result = speculative_generate(
        target,
        draft,
        prompt,
        max_new_tokens=max_new_tokens,
        draft_length=3,
        generator=torch.Generator().manual_seed(0),
    )

    print(f"target-only tokens:       {baseline.tolist()}")
    print(f"speculative tokens:       {result.token_ids.tolist()}")
    print(f"target-only model calls:  {max_new_tokens}")
    print(f"target verification calls:{result.target_verification_calls:>3}")
    print(f"draft acceptance rate:    {result.acceptance_rate:.3f}")
    print("Different seeded samples are expected; both follow the target distribution.")
    print("Call counts illustrate the algorithm, not measured wall-clock speed.")


if __name__ == "__main__":
    main()
