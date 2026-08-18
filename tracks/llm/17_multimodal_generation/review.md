# Review questions

- What work is performed once during prefill?
- Why is there one key/value cache pair per Transformer layer?
- Which cache dimension grows after each decoded token?
- Why does a one-token decode query use an all-true attention mask here?
- Why must the decoded token's position equal the previous cache length?
- How can you verify cached decoding is numerically correct?
- Why are vocabulary logits returned only for text positions?
- What extra state is needed to support padded prompts of different lengths?
- Why does this simple batched loop still compute finished EOS rows?
- How would sampling from `llm.10_sampling` replace greedy `argmax`?
