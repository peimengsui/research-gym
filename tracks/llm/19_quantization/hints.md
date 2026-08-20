# Hints

## Hint 1

Use `torch.where(max_absolute > 0, max_absolute / qmax,
torch.ones_like(max_absolute))` to preserve dtype and device.

## Hint 2

The quantization sequence is `round`, then `clamp`, then `.to(torch.int8)`.

## Hint 3

Dequantization is simply
`quantized_weight.to(scale.dtype) * scale`. Broadcasting handles row scales.

## Hint 4

For `(out_features, in_features)` weights, use
`weight.abs().amax(dim=1, keepdim=True)`.

## Hint 5

`WeightOnlyQuantizedLinear.from_float` should choose between
`quantize_per_output_channel` and `quantize_per_tensor`.

## Hint 6

Recursive replacement uses
`for name, child in list(module.named_children()):` followed by `setattr`.

## Hint 7

Model storage includes `list(module.parameters()) + list(module.buffers())`.
