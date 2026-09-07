# Hints

## Hint 1

Both action transforms are elementwise affine operations. Bounds with shape
`[action_dim]` broadcast across `[batch, chunk_size, action_dim]`.

## Hint 2

The fusion input is `torch.cat((video_features, language_features), dim=-1)`.

## Hint 3

Apply `torch.tanh` before reshaping the action-head output. Reshaping does not
change element order.

## Hint 4

Use `validity.unsqueeze(-1).expand_as(predictions)` to give each action
dimension the validity of its chunk step.

## Hint 5

`masked_select` turns the valid action errors into one vector whose mean has the
desired denominator.
