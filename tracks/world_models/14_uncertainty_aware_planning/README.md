# Model Disagreement and Safer Plans

Complete the foundational WAM sequence by separating two kinds of predictive
uncertainty. You will preserve valid multimodal outcomes within each stochastic
model, measure disagreement across an ensemble, and use only that epistemic
signal to penalize unfamiliar plans.

## What you will build

- expected future trajectories for every stochastic ensemble member
- an aleatoric and epistemic uncertainty decomposition
- nominal goal-and-effort plan scores
- epistemic-risk-adjusted scores
- uncertainty-aware candidate selection

The small WAM ensemble, candidate action chunks, controlled future predictions,
data containers, and input validation are provided. No model training, simulator,
or downloaded data is required.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
