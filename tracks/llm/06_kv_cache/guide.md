# Implementation guide

Open `implementation.py` in your workspace and fill in the TODOs.

## 1. Build a cached causal mask

For ordinary full-context attention, query length and key length are the same.
For cached attention, key length is larger:

```text
key_length = past_length + query_length
```

Query position `i` is allowed to attend through absolute key position:

```text
past_length + i
```

## 2. Append keys and values

In `CachedCausalSelfAttention.forward`:

```python
query = self.query(x)
new_key = self.key(x)
new_value = self.value(x)
```

If a cache exists, concatenate old and new keys/values along time dimension.
Return both the attended output and the updated cache.

## 3. Full-context path

`TinyCachedGPT.forward` should behave like the Tiny GPT lesson:

```text
token + position embeddings
-> blocks
-> final norm
-> language-model head
```

When targets are provided, compute cross-entropy over flattened batch and time.

## 4. Cached step path

`forward_step` receives only new tokens. Its positions start at
`position_offset`, and it receives one cache per block.

Each block returns an updated cache. Collect them in a list and return logits for
the new tokens.

## 5. Cached generation

First prefill the cache one prompt token at a time. Then repeatedly:

```text
run the newest token with the cache
take argmax of final logits
append token
```

This lesson limits cached generation to `block_size` total tokens so positional
embeddings stay simple.
