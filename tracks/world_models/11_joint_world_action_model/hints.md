# Hints

## Hint 1

Only the call to `video_encoder` belongs under `torch.no_grad()` in
`encode_video`; `state_encoder` must remain trainable for current observations.

## Hint 2

The action branch concatenates `state_latents` and `language_features`. The
dynamics branch concatenates `state_latents` and normalized transition actions.

## Hint 3

The flat action head has shape `[batch, chunk_size * action_dim]`. Apply `tanh`
before reshaping it to `[batch, chunk_size, action_dim]`.

## Hint 4

Use `state_latents + self.dynamics_model(dynamics_inputs)` for the residual
transition.

## Hint 5

Build the target with `with torch.no_grad(): target = self.encode_video(...)`.
The current state stays attached and receives gradients from both branches.

## Hint 6

Component losses are unweighted diagnostics. Apply `action_weight` and
`latent_weight` only to `total_loss`.
