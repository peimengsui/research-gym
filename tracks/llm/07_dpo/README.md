# Direct Preference Optimization

Direct Preference Optimization, or DPO, is a compact way to teach a language
model from pairwise preferences:

```text
same prompt
→ chosen completion
→ rejected completion
```

Instead of training a reward model and running reinforcement learning, DPO
compares how much more the current policy likes the chosen answer than the
rejected answer, relative to a frozen reference model.

In this lesson you will implement:

- token-level log-probability gathering
- completion log-probabilities for prompt/completion pairs
- the DPO loss
- one tiny preference-training step

The demo uses a toy CPU-friendly model and preference dataset. The point is the
equation and tensor flow, not chatbot-scale fine-tuning.

## Start

```bash
uv run rgym start llm.07_dpo
cd workspace/llm.07_dpo
uv run rgym test
uv run rgym run
```
