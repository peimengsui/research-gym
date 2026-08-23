# Video-Text Generation and Evaluation

This lesson completes the native-video sequence by using contextualized video
tubelets as a prefix for causal text prediction.

The full factorized encoder from `llm.22–23` is carried into `provided.py`. You
will implement:

- a dense video-prefix / causal-text attention mask
- multimodal self-attention and next-token training targets
- a tiny end-to-end video language model
- greedy batched generation with EOS handling
- candidate scoring and a small generation/ranking evaluation harness

Everything uses fixed-size synthetic clips, a tiny vocabulary, and CPU tensors.

## Start

```bash
uv run rgym start llm.24_tiny_video_language_model
cd workspace/llm.24_tiny_video_language_model
uv run rgym test
uv run rgym run
```
