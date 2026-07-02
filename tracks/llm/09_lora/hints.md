# Hints

## Hint 1

Freezing a layer means changing the parameters, not wrapping the forward pass in
`torch.no_grad()`.

## Hint 2

For a bias-free `nn.Linear(in_features, rank)`, the weight shape is:

```text
(rank, in_features)
```

## Hint 3

The LoRA delta weight should have the same shape as the base weight:

```python
lora_b.weight @ lora_a.weight
```

## Hint 4

If the merged layer does not match the unmerged layer, check whether you forgot
the `alpha / rank` scaling factor.
