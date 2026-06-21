# Concept: a bigram language model

A language model predicts the next token from tokens that came before it. A
bigram model uses the shortest possible context: only the current token.

If the vocabulary has `V` tokens, the model learns a `V × V` table. Row `i`
contains the logits for the token after token `i`. In PyTorch, an
`nn.Embedding(V, V)` is exactly this table:

```text
token ids [batch, time]
          ↓ embedding lookup
logits    [batch, time, vocab_size]
```

During training, cross-entropy compares each row of logits with the actual next
token. The model gradually learns local transitions such as which characters
often follow `r` or a space.

Generation is autoregressive:

1. run the current sequence through the model
2. select the logits at the final time step
3. apply softmax to get probabilities
4. sample one token
5. append it and repeat

This model cannot understand long-range context, but it contains the same basic
training and generation loop used by larger autoregressive language models.
