## Hint 1

Do not reshape `[batch, time]` into a flat dimension for this lesson. The GRU
needs the sequence order.

## Hint 2

Use `torch.cat((latents, actions), dim=-1)` before passing inputs to the GRU.

## Hint 3

The mean and log-sigma heads should end with `num_mixtures * latent_dim` values,
then reshape to `[batch, time, num_mixtures, latent_dim]`.

## Hint 4

Use `F.log_softmax(logits, dim=-1)` for mixture log probabilities.

## Hint 5

`torch.logsumexp` is the stable way to add probabilities in log space.
