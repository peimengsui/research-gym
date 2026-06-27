# Review checklist

- Can you trace `observation -> latent -> next latent -> next observation`?
- Did you keep tensor shapes explicit across batch, time, and channels?
- Does the dynamics model predict a residual delta?
- Does rollout use the model's own predicted latents after the first step?
- Can you explain why reconstruction and prediction losses are both useful?
