# Concept: exact speculative decoding

Autoregressive decoding normally invokes the target model once per generated
token. Speculative decoding adds a cheaper draft model that proposes a block of
future tokens. The target evaluates all proposed positions together.

## Draft and verify

For context `x`, the draft distribution `q` samples a candidate `y`. The target
distribution `p` assigns its own probability to that candidate. Accept with:

```text
acceptance_probability = min(1, p(y) / q(y))
```

If the candidate is at least as likely under the target as under the draft, it
is always accepted. Otherwise it is accepted only some of the time.

Drafting is autoregressive: after sampling each proposal, append it to the local
context before obtaining the next draft distribution. Verification returns a
target distribution for every proposal and one extra position.

## Correct a rejection

At the first rejected draft token, discard the remaining draft suffix and sample
one replacement from:

```text
normalize(max(p - q, 0))
```

Sampling directly from `p` again would double-count probability mass already
represented by accepted draft proposals. The positive residual correction is
what preserves the target distribution.

## Bonus token

If every draft token is accepted, sample one additional token from the target's
final returned distribution. A block of `gamma` proposals can therefore emit
`gamma + 1` tokens after one target verification call.

## Exact output, conditional efficiency

The output distribution matches ordinary target sampling within numerical
precision. Efficiency depends on draft cost, target parallelism, and acceptance
rate. This lesson counts verification calls but intentionally does not treat tiny
CPU timing as evidence of production speedup.
