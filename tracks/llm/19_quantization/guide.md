# Guide

Open `implementation.py`. Complete the eight TODOs in order.

## 1. Calculate one symmetric scale

Compute `max(abs(weight)) / qmax`. Keep the result as a scalar tensor on the
same device and with the same floating dtype. Use a scale of one when every
weight is zero.

## 2. Quantize one tensor

Divide by the scale, round to the nearest integer, clamp to the symmetric range,
and convert to `torch.int8`. Return both the compact values and scale because the
integer tensor alone cannot reconstruct approximate weights.

## 3. Dequantize with broadcasting

Convert int8 values to `scale.dtype` and multiply by scale. The same expression
works for a scalar scale and for `(out_features, 1)` row scales.

## 4. Quantize each output channel

For a 2D linear weight, compute maximum absolute values across dimension 1 with
`keepdim=True`. Handle all-zero rows independently. The resulting scale shape
must be `(out_features, 1)`.

## 5. Convert one linear layer

In `from_float`, quantize `layer.weight.detach()` with the selected mode and pass
the quantized weight, scale, and original bias to the constructor. The
constructor clones them into buffers.

## 6. Run the simulated forward

Dequantize the stored weight and call `F.linear(x, weight, self.bias)`. This is
readable and numerically useful, but it is not an optimized integer kernel.

## 7. Convert a module tree

Handle a root `nn.Linear` directly. Otherwise, recursively convert each named
child and install its replacement with `setattr`. Iterate over a list of named
children so mutation does not alter the iterator underneath you.

## 8. Measure persistent storage

For every parameter and buffer, add `numel() * element_size()`. Quantized weights
are buffers, while embeddings and normalization values remain parameters.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include using `-128` with `127` as an asymmetric symmetric range,
dividing a zero tensor by zero, forgetting `keepdim=True`, concatenating scales
with the wrong axis, leaving quantized weights as trainable parameters, counting
only parameters for storage, or claiming this simulated layer must run faster.
