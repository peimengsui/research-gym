# Token Rewards and Advantage Estimation

Turn generated language-model responses into the frozen training data needed by
policy-gradient methods. You will align response tokens with next-token model
outputs, preserve old-policy and reference log-probabilities, add a terminal
sequence score to per-token KL penalties, and compute masked advantages.

## What you will build

- target-aligned masks that select response tokens but exclude prompts/padding
- per-token next-token log-probabilities
- KL-shaped token rewards with a terminal deterministic score
- masked generalized advantage estimates and return targets
- a detached rollout batch ready for the next PPO lesson

A tiny causal policy, value model, greedy response generation, deterministic
toy reward, and input validation are provided. Generation is carried forward as
provided code from the earlier sampling lesson so the learner can focus on RL
data alignment rather than rewriting decoding.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
