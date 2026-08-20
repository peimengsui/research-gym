# Weight Quantization

Large language models store most of their parameters in linear-layer weight
matrices. This lesson replaces floating-point linear weights with symmetric int8
values and explicit floating-point scales.

You will implement:

- symmetric signed quantization scales
- per-tensor quantization and dequantization
- per-output-channel scales for linear weights
- an inference-only weight-quantized linear module
- recursive conversion of every linear layer in a tiny token model
- persistent parameter-and-buffer storage measurement

This is a transparent **simulated weight-only** implementation. Int8 weights are
dequantized before ordinary `F.linear` computation. It demonstrates storage and
approximation behavior but does not use optimized integer kernels or promise a
speedup.

## Start

```bash
uv run rgym start llm.19_quantization
cd workspace/llm.19_quantization
uv run rgym test
uv run rgym run
```
