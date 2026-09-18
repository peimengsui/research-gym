"""Run repeated PPO updates on one tiny frozen rollout."""

import copy
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from implementation import ppo_objective, ppo_update  # noqa: E402
from provided import TinyPPOModel, make_toy_rollout_batch  # noqa: E402


def main() -> None:
    torch.manual_seed(32)
    old_model = TinyPPOModel(vocab_size=10, hidden_dim=16)
    rollout = make_toy_rollout_batch(old_model)
    current_model = copy.deepcopy(old_model)
    optimizer = torch.optim.Adam(current_model.parameters(), lr=0.01)
    parameters_before = torch.cat(
        [parameter.detach().flatten() for parameter in current_model.parameters()]
    )

    _, initial = ppo_objective(current_model, rollout)
    history = ppo_update(
        current_model,
        rollout,
        optimizer,
        num_epochs=3,
        minibatch_size=2,
        generator=torch.Generator().manual_seed(32),
    )
    _, final = ppo_objective(current_model, rollout)
    parameters_after = torch.cat(
        [parameter.detach().flatten() for parameter in current_model.parameters()]
    )

    print(f"response tokens:       {rollout.response_mask.sum().item()}")
    print(f"minibatch updates:      {len(history)}")
    print(f"initial policy loss:    {initial.policy_loss.item():.6f}")
    print(f"final policy loss:      {final.policy_loss.item():.6f}")
    print(f"initial value loss:     {initial.value_loss.item():.6f}")
    print(f"final value loss:       {final.value_loss.item():.6f}")
    print(f"final approximate KL:   {final.approx_kl.item():.6f}")
    print(f"final clip fraction:    {final.clip_fraction.item():.3f}")
    print(
        f"parameter change norm:  {(parameters_after - parameters_before).norm():.6f}"
    )
    print("Frozen old-policy statistics were reused across every PPO update.")


if __name__ == "__main__":
    main()
