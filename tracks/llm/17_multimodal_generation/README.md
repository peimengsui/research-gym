# Image Prefill, KV Cache, and Decoding

A vision-language model should not re-encode the same image after every token it
generates. This lesson splits inference into a single multimodal **prefill** and
small one-token **decode** steps that reuse per-layer key/value caches.

You will implement:

- the dense-visual-prefix and causal-text attention mask
- multi-head key/value cache creation and append behavior
- one image-and-prompt prefill pass
- one-token decoding against cached image and text positions
- greedy generation with optional EOS stopping

The vision patch embedding, feed-forward layer, tiny vocabulary, and synthetic
images are supplied in `provided.py`. Prompts have equal lengths and contain no
padding so the exercise can focus on cache mechanics. Production systems also
track per-row prompt validity and compact finished sequences.

## Start

```bash
uv run rgym start llm.17_multimodal_generation
cd workspace/llm.17_multimodal_generation
uv run rgym test
uv run rgym run
```
