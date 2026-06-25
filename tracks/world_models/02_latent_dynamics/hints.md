## Hint 1

Slice `latents[:, :-1, :]` for current states and `latents[:, 1:, :]` for next
states.

## Hint 2

After slicing trajectories, flatten batch and time together with `reshape`.

## Hint 3

Concatenate latent state and action with `torch.cat((z, action), dim=-1)`.

## Hint 4

The network predicts a delta. The final next-state prediction should be
`z + delta`.

## Hint 5

For rollout, keep replacing `current_z` with the model's newest prediction.
That is what makes the rollout autoregressive.
