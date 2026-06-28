# Implementation guide

Open `implementation.py` in your workspace and fill in the TODOs.

The Transformer block pieces are included so you can focus on assembling GPT.

## 1. Build language-model batches

Given a 1D token tensor, sample starting positions. For each start:

```text
input  = data[start : start + block_size]
target = data[start + 1 : start + block_size + 1]
```

Raise `ValueError` if the data is not 1D, `block_size` or `batch_size` is not
positive, or the data is not longer than `block_size`.

## 2. Add embeddings and layers

In `TinyGPT.__init__`, create:

```python
self.token_embedding = nn.Embedding(vocab_size, embed_dim)
self.position_embedding = nn.Embedding(block_size, embed_dim)
self.blocks = nn.Sequential(
    *[TransformerBlock(embed_dim, expansion_factor) for _ in range(num_layers)]
)
self.final_norm = nn.LayerNorm(embed_dim)
self.lm_head = nn.Linear(embed_dim, vocab_size)
```

Also validate that `vocab_size`, `block_size`, `embed_dim`, and `num_layers` are
positive.

## 3. Implement the forward pass

`idx` has shape:

```text
[batch, time]
```

Reject sequences longer than `block_size`.

Create positions with:

```python
positions = torch.arange(time, device=idx.device)
```

Then:

```text
token embeddings + position embeddings
-> blocks
-> final norm
-> lm head
```

If targets are provided, compute cross-entropy by flattening batch and time.

## 4. Implement generation

For each new token:

```python
context = idx[:, -self.block_size :]
logits, _ = self(context)
final_logits = logits[:, -1, :]
probabilities = F.softmax(final_logits, dim=-1)
next_token = torch.multinomial(probabilities, num_samples=1)
idx = torch.cat((idx, next_token), dim=1)
```

Cropping the context is what allows generation to continue beyond `block_size`.
