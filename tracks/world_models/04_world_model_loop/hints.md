## Hint 1

This lesson keeps time in the tensors. Do not flatten `[batch, time]` away.

## Hint 2

`nn.Linear` works on the final dimension, so your MLPs can accept
`[batch, time, channels]` tensors directly.

## Hint 3

The dynamics network predicts a delta. Add it back to the current latent.

## Hint 4

For rollout, keep replacing `current_latent` with the newest predicted latent.

## Hint 5

The latent target should be `target_next_latents.detach()` inside the loss.
That prevents the target encoder path from chasing the prediction.
