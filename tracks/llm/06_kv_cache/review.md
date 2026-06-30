# Review checklist

- Can you explain why keys and values can be cached but queries cannot?
- Does cached decoding produce the same greedy tokens as full-context decoding?
- Did each Transformer block maintain its own cache?
- Did `forward_step` use the correct positional offset?
- Why does a KV cache matter more for long contexts than tiny demos?
