# Concept: MDN-RNN

A deterministic dynamics model predicts one next latent state. An MDN-RNN
predicts a distribution over possible next latent states.

MDN means mixture density network. Instead of outputting only `z_hat_{t+1}`, the
model outputs parameters for several Gaussian components:

```text
mixture logits: which component is likely?
means:          where is each component centered?
log sigmas:     how wide is each component?
```

RNN means recurrent neural network. The GRU hidden state carries information
from earlier time steps:

```text
input_t = concat(z_t, a_t)
hidden_t = GRU(input_t, hidden_{t-1})
mixture_t = heads(hidden_t)
```

## Why keep time?

The deterministic latent-dynamics lesson flattened `[batch, time]` into one
supervised dimension. An RNN needs the order, so this lesson keeps tensors in
sequence form:

```text
[batch, time, channels]
```

## Why negative log likelihood?

The target next latent should be likely under the predicted mixture. The loss is
the negative log probability of the target under all mixture components. Lower
loss means the model assigned higher probability to the observed future.

This is the bridge from one-step prediction toward richer imagined futures.
