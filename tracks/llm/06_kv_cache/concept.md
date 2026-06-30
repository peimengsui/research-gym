# Concept: KV cache

In a GPT-style model, each attention layer projects token representations into:

```text
query, key, value
```

During generation, previous tokens do not change. Their keys and values do not
change either. A KV cache stores them.

## Full-context decoding

Without a cache, generation does this every step:

```text
run the whole context -> take final logits -> sample next token
```

That is simple, but it repeatedly recomputes keys and values for old tokens.

## Cached decoding

With a cache:

```text
run only the newest token
append its key/value to the cache
attend to cached past + new token
take final logits
```

The logits should match full-context decoding. The cache changes how much work
is repeated, not what the model computes.

This lesson uses greedy decoding so equality is easy to test.
