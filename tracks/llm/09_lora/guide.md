# Guide

Open `implementation.py`. The main object is `LoRALinear`.

## 1. Freeze the base layer

`LoRALinear` owns a normal `nn.Linear` called `base`.

Every parameter in `base` should have:

```python
requires_grad = False
```

That is the heart of parameter-efficient tuning: the original model stays fixed.

## 2. Add low-rank matrices

The adapter has two bias-free linear layers:

```text
lora_a: in_features -> rank
lora_b: rank -> out_features
```

The adapter path should compute:

```python
lora_b(lora_a(x)) * scaling
```

where:

```python
scaling = alpha / rank
```

## 3. Initialize carefully

Initialize `lora_a.weight` with a small random distribution and
`lora_b.weight` to zeros.

At initialization, the full layer should behave exactly like the frozen base
layer because the LoRA branch is zero.

## 4. Count parameters

`count_trainable_parameters(module)` should count only parameters where
`requires_grad` is `True`.

For LoRA, that should be much smaller than the total parameter count.

## 5. Merge for inference

The matrix update has shape:

```text
lora_b.weight @ lora_a.weight
```

That gives:

```text
(out_features, in_features)
```

which matches the base linear weight. The merged weight is:

```python
base.weight + scaling * (lora_b.weight @ lora_a.weight)
```

## Run

```bash
uv run rgym test
uv run rgym run
```
