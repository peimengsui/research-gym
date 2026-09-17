"""Generate tiny responses and inspect their frozen RL rollout statistics."""

import copy
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import (  # noqa: E402
    collect_rollout_batch,
    response_mask_from_prompt_lengths,
)
from provided import (  # noqa: E402
    TinyRolloutLM,
    TinyValueModel,
    deterministic_response_scores,
    greedy_generate,
)


def main() -> None:
    torch.manual_seed(31)
    vocab_size = 12
    old_policy = TinyRolloutLM(vocab_size, hidden_dim=16)
    reference_policy = copy.deepcopy(old_policy)
    with torch.no_grad():
        reference_policy.lm_head.weight.add_(
            0.04 * torch.randn_like(reference_policy.lm_head.weight)
        )
    value_model = TinyValueModel(vocab_size, hidden_dim=16)
    prompts = torch.tensor([[1, 2], [1, 3], [1, 4]])
    generated = greedy_generate(old_policy, prompts, max_new_tokens=4)
    response_mask = response_mask_from_prompt_lengths(
        generated.input_ids,
        generated.attention_mask,
        generated.prompt_lengths,
    )
    scores = deterministic_response_scores(generated.input_ids, response_mask)
    rollout = collect_rollout_batch(
        old_policy,
        reference_policy,
        value_model,
        generated.input_ids,
        generated.attention_mask,
        generated.prompt_lengths,
        scores,
        kl_coefficient=0.1,
        gamma=1.0,
        gae_lambda=0.95,
    )

    response_tokens = rollout.response_mask.sum().item()
    mean_kl = rollout.token_kl[rollout.response_mask].mean().item()
    reward_sums = rollout.token_rewards.sum(dim=1)
    mean_advantage = rollout.advantages[rollout.response_mask].mean().item()
    print(f"generated sequences:\n{rollout.input_ids}")
    print(f"response mask:\n{rollout.response_mask}")
    print(f"deterministic scores: {rollout.scores.tolist()}")
    print(f"response tokens:      {response_tokens}")
    print(f"mean sampled KL:      {mean_kl:.6f}")
    print(f"shaped reward sums:   {reward_sums.tolist()}")
    print(f"mean advantage:       {mean_advantage:.6f}")
    print(f"rollout requires grad: {rollout.old_logprobs.requires_grad}")


if __name__ == "__main__":
    main()
