# Hints

## Hint 1

For image tensors, reshape times to `(B, 1, 1, 1)`.

## Hint 2

The target velocity points from the noise endpoint toward the data endpoint.
Swapping the subtraction reverses the generation direction.

## Hint 3

Flow-matching loss compares velocities. It does not compare predicted noise or
predicted clean samples.

## Hint 4

Euler uses one model evaluation per step. Midpoint uses two.

## Hint 5

At ODE step `i`, the starting time is `i / num_steps` and the step size is
`1 / num_steps`.

## Hint 6

A returned trajectory has `num_steps + 1` states because it includes the
initial Gaussian noise.
