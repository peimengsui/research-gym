# Reflection questions

- Why can `nn.Embedding(vocab_size, vocab_size)` represent a bigram model?
- Why must logits and targets be flattened before cross-entropy?
- Why does generation use only the final time step's logits?
- What kinds of language patterns can this model learn, and what can it not
  learn?
- How would increasing the context length change the model?
