# Receding-Horizon Planning with a WAM

Turn a joint world-action model into a controller. At every environment step,
you will imagine candidate action chunks in latent space, score their predicted
goal progress, execute only the first action from the best chunk, and replan
from a fresh observation.

## What you will build

- batched latent rollouts for candidate action chunks
- a goal-progress score with a small action-effort penalty
- language-conditioned action-chunk selection using provided CEM search
- a receding-horizon observe-plan-act loop

The tiny continuous grid, video renderer, language-goal encoder, exact frozen
WAM, input validation, and CEM optimizer mechanics are provided. The exact model
stands in for a converged `wm.11` so this lesson can isolate planning behavior.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
