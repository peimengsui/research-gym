# Review checklist

- Can you explain the residual stream in your own words?
- Did the attention and MLP sublayers both preserve `[batch, time, embed_dim]`?
- Did you apply LayerNorm before each sublayer?
- Can you explain why the MLP does not mix information across token positions?
- Can you describe how this block could be stacked to build a tiny GPT?
