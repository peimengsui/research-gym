# Hints

## Hint 1

Use one shared encoder hidden state, then two separate linear heads for
`mu` and `logvar`.

## Hint 2

`logvar` stores the logarithm of the variance. Convert it to standard deviation
with `torch.exp(0.5 * logvar)`.

## Hint 3

Use `torch.randn_like(std)` for `epsilon`; this automatically matches shape,
device, and dtype.

## Hint 4

The decoder should end with sigmoid when reconstruction uses binary
cross-entropy.

## Hint 5

Sum both losses and divide by batch size. The KL term is
`-0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size`.
