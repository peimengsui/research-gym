# Review questions

- Why does this symmetric int8 scheme leave the `-128` code unused?
- Why must an all-zero tensor receive a nonzero scale?
- Where does rounding error enter quantization?
- When should per-channel scales outperform one per-tensor scale?
- Why are linear output rows a natural channel axis?
- Why are quantized weights registered as buffers instead of parameters?
- Can gradients still reach the input of a weight-only quantized layer?
- Why must storage accounting include buffers?
- Why does smaller persistent storage not guarantee faster execution here?
- What would optimized kernels, activation quantization, or calibration add?
