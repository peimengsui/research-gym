# ResearchGym roadmap

The roadmap favors a sequence of small implementations that build conceptual
and code-level fluency. Ordering is directional rather than a release promise.

## Current foundation

- CLI lesson discovery and inspection
- isolated learner workspaces
- test, demo, hint, and report commands
- complete Bigram Language Model lesson
- complete Tokenizer Fundamentals lesson
- complete Causal Self-Attention lesson
- complete Transformer Block lesson
- complete Tiny GPT lesson
- complete KV Cache lesson
- complete Direct Preference Optimization lesson
- complete Supervised Fine-Tuning Data lesson
- complete Low-Rank Adaptation lesson
- complete Language Model Sampling lesson
- complete Images as Patch Tokens lesson
- complete Visual Transformer Blocks lesson
- complete Unified Image and Text Tokens lesson
- complete Visual Prefix and Causal Text lesson
- complete Tiny Native Vision-Language Model lesson
- complete Variational Autoencoder lesson
- complete Latent Dynamics lesson
- complete MDN-RNN lesson
- complete World Model Loop lesson
- complete CEM Planning lesson
- complete Noise Schedules and the Forward Process lesson
- complete Epsilon Prediction Objective lesson
- complete Reverse DDPM Sampling lesson
- complete Tiny U-Net Denoiser lesson
- complete Deterministic DDIM Sampling lesson

## Track roadmaps

These sequences are directional. A lesson can be split, renamed, or delayed if
the implementation stops feeling small, CPU-friendly, and easy to inspect.

### Language models

Completed:

- `llm.01_bigram_lm` — Bigram Language Model
- `llm.02_tokenizer` — Tokenizer Fundamentals
- `llm.03_causal_attention` — Causal Self-Attention
- `llm.04_transformer_block` — Transformer Block
- `llm.05_tiny_gpt` — Tiny GPT
- `llm.06_kv_cache` — KV Cache
- `llm.07_dpo` — Direct Preference Optimization
- `llm.08_sft_data` — Supervised Fine-Tuning Data
- `llm.09_lora` — Low-Rank Adaptation
- `llm.10_sampling` — Language Model Sampling
- `llm.11_vision_patch_embeddings` — Images as Patch Tokens
- `llm.12_vision_attention` — Visual Transformer Blocks
- `llm.13_multimodal_sequence` — Unified Image and Text Tokens
- `llm.14_multimodal_attention_mask` — Visual Prefix and Causal Text
- `llm.15_tiny_native_vlm` — Tiny Native Vision-Language Model

Planned:

The next sequence continues native vision support. Image patches become model
tokens and eventually share Transformer layers with text tokens, rather than
depending on an opaque pretrained vision encoder. Lessons should use synthetic
Torch images so tests remain self-contained and CPU-friendly.

```text
image patch tokens
→ visual attention
→ unified image-text sequence
→ multimodal attention mask
→ tiny native VLM
→ multimodal training, generation, and evaluation
```

- `llm.16_multimodal_sft_data` — Image-Text Conversations
- `llm.17_multimodal_generation` — Image Prefill, KV Cache, and Decoding
- `llm.18_multimodal_eval` — Tiny Vision-Language Evaluation Harness
- `llm.19_quantization` — Weight Quantization

### World models

Completed:

- `wm.01_vae` — Variational Autoencoder
- `wm.02_latent_dynamics` — Latent Dynamics
- `wm.03_mdn_rnn` — MDN-RNN
- `wm.04_world_model_loop` — World Model Loop
- `wm.05_cem_planning` — CEM Planning

Planned:

- `wm.06_dreamer_lite` — Latent Imagination and Actor-Critic Components
- `wm.07_uncertainty` — Ensembles and Model Uncertainty
- `wm.08_policy_in_latent_space` — Policy Learning Inside a Learned World

### Diffusion models

The diffusion track should start with tiny tensor exercises before moving to
image-shaped models. It is informed by Peimeng Sui's diffusion self-study notes:
<https://peimengsui.github.io/2026/01/25/Diffusion-Learning-Summary.html>.

Completed:

- `diffusion.01_forward_process` — Noise Schedules and the Forward Process
- `diffusion.02_noise_prediction` — Epsilon Prediction Objective
- `diffusion.03_ddpm_sampling` — Reverse DDPM Sampling
- `diffusion.04_tiny_unet` — Tiny U-Net Denoiser
- `diffusion.05_ddim_sampling` — Deterministic DDIM Sampling
- `diffusion.06_classifier_free_guidance` — Conditional Generation and Guidance
- `diffusion.07_latent_diffusion` — Diffusion in a Learned Latent Space
- `diffusion.08_cross_attention_conditioning` — Prompt-Style Conditioning
- `diffusion.09_flow_matching` — Flow Matching and ODE Sampling

Remaining plan:

Stage 3 — modern extensions:

- `diffusion.10_consistency_models` — One/Few-Step Consistency Models

## Implementation stages

### Stage 1: continue core tracks

### Language models

- continue native vision with `llm.16_multimodal_sft_data`

### World models

- Dreamer-lite latent imagination and actor-critic components

### Diffusion models

- continue with `diffusion.10_consistency_models`

### Stage 2: deepen track coverage

- native image patch tokens, visual attention, and unified image-text sequences
- world-model uncertainty and policy learning in latent space
- diffusion noise prediction, DDPM sampling, and tiny U-Net denoising

### Stage 3: connect to broader research patterns

- small vision-language model lab
- multimodal training data, generation, and evaluation
- tiny diffusion image lab
- lesson authoring validation and contribution templates
- richer reports that summarize tests, demos, and learner reflection

## Guiding constraints

New lessons should remain:

- understandable from source
- runnable on CPU for tests
- small enough to complete incrementally
- explicit about tensor shapes and objectives
- independent of notebooks, distributed training, and hosted experiment tools
