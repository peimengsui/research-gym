# Low-Rank Adaptation

LoRA, short for Low-Rank Adaptation, is a parameter-efficient fine-tuning
method. Instead of updating a large pretrained weight matrix, LoRA freezes the
base weights and trains a small low-rank update.

For a linear layer:

```text
y = x W^T + b
```

LoRA adds:

```text
y = x W^T + scale * x (B A)^T + b
```

where:

- `W` is frozen
- `A` maps from input dimension to a small rank
- `B` maps from that rank to output dimension
- only `A` and `B` are trainable

In this lesson you will implement:

- a `LoRALinear` module
- frozen base weights
- trainable low-rank adapter weights
- parameter counting
- merging LoRA weights back into a normal linear layer

## Start

```bash
uv run rgym start llm.09_lora
cd workspace/llm.09_lora
uv run rgym test
uv run rgym run
```
