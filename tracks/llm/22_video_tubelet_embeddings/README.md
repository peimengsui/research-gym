# Videos as Spatiotemporal Tokens

An image patch covers a small spatial region. A video tubelet extends that patch
through several neighboring frames, so one token contains local appearance and
short-term motion.

You will implement:

- tubelet extraction from `(batch, frames, channels, height, width)` videos
- exact reconstruction back to the original clip
- linear projection from flattened tubelets to model embeddings
- separate learned temporal and spatial position embeddings

The lesson uses tiny synthetic clips and runs entirely on CPU.

## Start

```bash
uv run rgym start llm.22_video_tubelet_embeddings
cd workspace/llm.22_video_tubelet_embeddings
uv run rgym test
uv run rgym run
```
