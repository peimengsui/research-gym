"""Train a tiny policy with Direct Preference Optimization."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        TinyPreferenceLM,
        completion_logprobs,
        make_toy_preference_data,
        train_dpo_step,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        TinyPreferenceLM,
        completion_logprobs,
        make_toy_preference_data,
        train_dpo_step,
    )


def preference_margin(
    model: TinyPreferenceLM,
    prompts: torch.Tensor,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
) -> torch.Tensor:
    chosen_logps = completion_logprobs(model, prompts, chosen)
    rejected_logps = completion_logprobs(model, prompts, rejected)
    return (chosen_logps - rejected_logps).mean()


def main() -> None:
    torch.manual_seed(7)
    prompts, chosen, rejected = make_toy_preference_data()
    policy = TinyPreferenceLM(vocab_size=12, hidden_dim=16)
    reference = TinyPreferenceLM(vocab_size=12, hidden_dim=16)
    reference.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.08)

    with torch.no_grad():
        initial_margin = preference_margin(policy, prompts, chosen, rejected)
    print(f"initial chosen-vs-rejected margin: {initial_margin.item():.3f}")

    for step in range(1, 61):
        stats = train_dpo_step(
            policy,
            reference,
            optimizer,
            prompts,
            chosen,
            rejected,
            beta=0.5,
        )
        if step in {1, 10, 30, 60}:
            print(
                f"step {step:02d} | loss={stats.loss.item():.3f} "
                f"| policy_margin={stats.policy_margin.item():.3f} "
                f"| preference_acc={stats.preference_accuracy.item():.2f}"
            )

    with torch.no_grad():
        final_margin = preference_margin(policy, prompts, chosen, rejected)
    print(f"final chosen-vs-rejected margin:   {final_margin.item():.3f}")
    print("DPO pushed the policy toward the chosen completions.")


if __name__ == "__main__":
    main()
