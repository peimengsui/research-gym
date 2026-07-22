# Classifier-Free Guidance

Classifier-free guidance (CFG) strengthens a diffusion model's response to a
condition without training a separate classifier. One noise predictor learns
both conditional and unconditional behavior.

In this lesson you will implement:

- class and null-condition embeddings in a tiny U-Net
- random condition dropout during training
- the classifier-free guidance equation
- paired conditional/unconditional model calls
- guided deterministic DDIM sampling

The scaffold carries forward the forward process and DDIM sampler. Your work is
focused on the new conditioning and guidance logic.

## Start

```bash
uv run rgym start diffusion.06_classifier_free_guidance
cd workspace/diffusion.06_classifier_free_guidance
uv run rgym test
uv run rgym run
```
