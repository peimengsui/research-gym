"""Run tiny GRPO updates over two groups of sampled responses."""

import copy
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    expand_response_advantages,
    grpo_objective,
    grpo_update,
    group_relative_advantages,
)
from provided import TinyGRPOPolicy, make_toy_grouped_rollout  # noqa: E402


def main() -> None:
    torch.manual_seed(33)
    old_policy = TinyGRPOPolicy(vocab_size=10, hidden_dim=16)
    rollout = make_toy_grouped_rollout(old_policy)
    current_policy = copy.deepcopy(old_policy)
    response_advantages = group_relative_advantages(
        rollout.scores,
        rollout.group_ids,
    )
    token_advantages = expand_response_advantages(
        response_advantages,
        rollout.response_mask,
    )
    optimizer = torch.optim.Adam(current_policy.parameters(), lr=0.01)

    _, initial = grpo_objective(current_policy, rollout, token_advantages)
    history = grpo_update(
        current_policy,
        rollout,
        optimizer,
        num_epochs=3,
        minibatch_size=2,
        generator=torch.Generator().manual_seed(33),
    )
    _, final = grpo_objective(current_policy, rollout, token_advantages)

    for group_id in torch.unique(rollout.group_ids):
        group_mask = rollout.group_ids == group_id
        scores = rollout.scores[group_mask].tolist()
        advantages = response_advantages[group_mask].tolist()
        print(f"group {group_id.item()} scores:      {scores}")
        print(
            f"group {group_id.item()} advantages:  {[round(x, 3) for x in advantages]}"
        )
    print(f"minibatch updates:       {len(history)}")
    print(f"initial policy loss:     {initial.policy_loss.item():.6f}")
    print(f"final policy loss:       {final.policy_loss.item():.6f}")
    print(f"final reference KL:      {final.reference_kl.item():.6f}")
    print(f"final old-policy KL:     {final.old_policy_kl.item():.6f}")
    print(f"final clip fraction:     {final.clip_fraction.item():.3f}")
    print("Group advantages were frozen before row-minibatch optimization.")


if __name__ == "__main__":
    main()
