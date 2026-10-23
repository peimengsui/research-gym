# First-Frame and Text-Guided Video Diffusion

This lesson adds two conditions to tiny video diffusion:

- a supplied first frame that must remain exactly preserved
- a tiny text instruction such as `move right` or `move up`

The spatiotemporal architecture, text encoder, cross-attention, classifier-free
guidance formula, training objective, and DDIM equation are provided. You will
implement diffusion-consistent first-frame clamping, the conditioned sampling
loop, and a direction-alignment diagnostic.

## Learning goals

- keep known pixels on the correct noisy trajectory during reverse diffusion
- separate hard first-frame constraints from soft text guidance
- integrate conditional and text-unconditional predictions with CFG
- guarantee that the final sample reproduces the supplied first frame
- measure whether generated motion follows a requested direction

## Run the lesson

```bash
uv run rgym start diffusion.16_conditioned_video_diffusion
cd workspace/diffusion.16_conditioned_video_diffusion
uv run rgym test
uv run rgym run
```

The demo trains four tiny text directions and samples two clips from the same
first frame and initial noise so the effect of the instruction is visible.
