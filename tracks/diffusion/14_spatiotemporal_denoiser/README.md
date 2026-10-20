# Spatiotemporal Video Denoiser

This lesson extends a small image denoiser to video tensors with shape
`(batch, channels, frames, height, width)`. Provided spatial residual blocks
process each frame independently. You will implement the shape transformations,
temporal convolution, complete factorized denoiser, and epsilon-prediction
objective.

The lesson does not implement a video sampling loop yet. Its purpose is to make
the denoiser's data path explicit and verify that information can travel both
within a frame and between neighboring frames.

## Learning goals

- move safely between video tensors and batches of image frames
- apply the same spatial block independently to every frame
- mix each spatial location along the temporal axis
- broadcast one diffusion timestep across every frame of a video
- train a video-shaped epsilon predictor with mean-squared error

## Run the lesson

```bash
uv run rgym start diffusion.14_spatiotemporal_denoiser
cd workspace/diffusion.14_spatiotemporal_denoiser
uv run rgym test
uv run rgym run
```

Edit `implementation.py`. The forward diffusion utilities and spatial block are
already available in `provided.py`, so the TODOs focus on the new video logic.
