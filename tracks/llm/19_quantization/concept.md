# Concept: symmetric weight quantization

Neural-network weights are usually stored as floating-point numbers. Quantization
approximates them with a smaller integer representation plus scale metadata.

## Symmetric mapping

For signed `b`-bit symmetric quantization, this lesson uses:

```text
qmax = 2^(b - 1) - 1
integer range = [-qmax, qmax]
```

Int8 therefore uses `[-127, 127]`. The `-128` code is left unused so positive
and negative magnitudes have matching ranges.

Given floating weights `w`, choose:

```text
scale = max(abs(w)) / qmax
q = clamp(round(w / scale), -qmax, qmax)
w_approx = q * scale
```

A completely zero tensor receives scale one. Its quantized values remain zero,
and the safe nonzero scale avoids division by zero.

## Per-tensor versus per-channel

Per-tensor quantization shares one scale across the entire weight matrix. One
large outlier can make the scale too coarse for small values elsewhere.

A linear weight has shape `(out_features, in_features)`. Per-output-channel
quantization computes one scale for each row, producing scales shaped
`(out_features, 1)`. Broadcasting applies each row's scale independently.

Per-channel metadata costs a few more floating-point values but usually reduces
reconstruction error when rows have different ranges.

## Weight-only linear inference

`WeightOnlyQuantizedLinear` stores:

- an int8 weight buffer
- a scalar or per-row floating scale buffer
- an optional floating bias buffer

Its forward pass dequantizes the weight and calls `F.linear`. The layer is
inference-only: quantized weights are buffers, not trainable parameters. Input
gradients still exist because the dequantized weight acts as a constant linear
mapping.

## Conversion and storage

Recursive conversion replaces every `nn.Linear` child while leaving embeddings,
normalization parameters, and other modules floating point. Counting only model
parameters after conversion would incorrectly report zero bytes for quantized
weights, because they are buffers. A useful storage estimate must count both.

## What this lesson does not claim

Persistent weight storage is smaller, but this implementation creates a
floating approximation during every forward call. It is intended to expose the
math, not to benchmark production inference. Actual acceleration requires
hardware-aware packed formats and integer or low-precision kernels. Activation
quantization, calibration, outlier handling, and quantization-aware training are
also outside this lesson's scope.
