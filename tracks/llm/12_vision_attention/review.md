# Review questions

- Why must `embed_dim` be divisible by `num_heads`?
- Which axes are transposed when splitting and merging heads?
- What do the final two axes of an attention map represent?
- Why does each attention row sum to one?
- Why is a causal mask appropriate for GPT but not for this visual encoder?
- What different relationships might separate heads learn between image patches?
- Why are layer normalization and residual connections arranged as pre-norm?
- Why does this encoder preserve every patch token instead of pooling them?
