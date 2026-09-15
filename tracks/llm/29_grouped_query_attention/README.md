# Shared Key-Value Heads and Smaller Caches

Keep many expressive query heads while sharing fewer key and value heads across
query groups. You will expand compact KV tensors only for attention computation,
preserve compact caches during decoding, and measure the resulting memory
reduction.

## What you will build

- mapping from compact KV heads to consecutive query-head groups
- grouped-query scaled dot-product attention
- exact key-plus-value cache element accounting
- RoPE causal GQA with incremental decoding

The RoPE implementation from `llm.28`, causal masking, attention primitive, head
reshaping, projection setup, and validation are provided. This keeps the lesson
focused on head sharing rather than repeating rotary-position mechanics.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
