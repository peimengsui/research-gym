# Concept: Low-Rank Adaptation

Full fine-tuning updates every parameter in a model. For large LLMs, that can be
expensive because every weight receives gradients and every updated checkpoint
stores another full copy of the model.

LoRA makes a different bet:

```text
keep the pretrained weight frozen
learn a small low-rank update beside it
```

For one linear layer, the frozen base path is:

```text
base(x) = x W^T + b
```

LoRA adds two smaller matrices:

```text
A: rank × in_features
B: out_features × rank
```

The update is:

```text
delta(x) = scale * x (B A)^T
```

So the final output is:

```text
base(x) + delta(x)
```

The rank is much smaller than the input/output dimensions, so the adapter has
far fewer trainable parameters than the full weight matrix.

## Why initialize `B` to zero?

Many LoRA implementations initialize `A` randomly and `B` to zero. That means
the adapter initially contributes exactly zero:

```text
LoRA output at step 0 = frozen base output
```

Training can then move away from the base model smoothly.

## Why merge?

At inference time, LoRA can be merged into the base matrix:

```text
W_merged = W + scale * B A
```

After merging, the model can run as a normal linear layer with no adapter branch.
This lesson implements that merge explicitly so the tensor shapes are visible.
