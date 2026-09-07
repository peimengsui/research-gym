# Review checklist

- Can you derive both action-coordinate transforms and show they are inverses?
- Does each action dimension use its own physical bounds?
- Where do video and language first interact in the policy?
- Why is `tanh` applied in normalized rather than environment coordinates?
- Does the output have shape `[batch, chunk_size, action_dim]`?
- Do padded chunk steps contribute neither numerator nor denominator to loss?
- Do gradients reach the visual encoder, language encoder, and action head?
- Why is this policy reactive rather than an explicit world model?
- What capability will a joint next-latent objective add in `wm.11`?
