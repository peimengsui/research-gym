## Hint 1

The hidden size of the feed-forward network is `embed_dim * expansion_factor`.

## Hint 2

`nn.Linear` can process `[batch, time, channels]` tensors directly. It transforms
only the final dimension.

## Hint 3

Pre-norm means `LayerNorm` happens before the attention or MLP sublayer, not
after the residual addition.

## Hint 4

Each sublayer is added back to `x`. That is the residual connection.

## Hint 5

Always ask: "Does this operation preserve `[batch, time, embed_dim]`?" If yes,
you are probably on the right track.
