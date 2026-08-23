# Draft, Verify, and Correct

Speculative decoding asks a cheap draft model to propose several tokens, then
uses one target-model verification pass to accept a prefix of those proposals.
A rejection correction preserves the target model's sampling distribution.

You will implement:

- draft-token acceptance probabilities
- the rejection correction distribution
- autoregressive block drafting
- one target verification step with a bonus token
- speculative generation, EOS handling, and acceptance statistics

`provided.py` contains tiny target and draft bigram probability models. They make
the distribution-preservation rule directly testable without a large model or
GPU. Target-call counts illustrate algorithmic reuse; this CPU lesson does not
claim a measured latency improvement.

## Start

```bash
uv run rgym start llm.20_speculative_decoding
cd workspace/llm.20_speculative_decoding
uv run rgym test
uv run rgym run
```
