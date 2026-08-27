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
- complete Image-Text Conversations lesson
- complete Image Prefill, KV Cache, and Decoding lesson
- complete Tiny Vision-Language Evaluation Harness lesson
- complete Weight Quantization lesson
- complete Draft, Verify, and Correct lesson
- complete KV Blocks and Request Scheduling lesson
- complete Videos as Spatiotemporal Tokens lesson
- complete Spatial and Temporal Attention lesson
- complete Video-Text Generation and Evaluation lesson
- complete Waveforms, STFT, and Audio Patches lesson
- complete Variable-Duration Audio and Attention lesson
- complete Audio-Text Generation and Evaluation lesson
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
- `llm.16_multimodal_sft_data` — Image-Text Conversations
- `llm.17_multimodal_generation` — Image Prefill, KV Cache, and Decoding
- `llm.18_multimodal_eval` — Tiny Vision-Language Evaluation Harness
- `llm.19_quantization` — Weight Quantization
- `llm.20_speculative_decoding` — Draft, Verify, and Correct
- `llm.21_paged_kv_and_continuous_batching` — KV Blocks and Request Scheduling
- `llm.22_video_tubelet_embeddings` — Videos as Spatiotemporal Tokens
- `llm.23_factorized_video_attention` — Spatial and Temporal Attention
- `llm.24_tiny_video_language_model` — Video-Text Generation and Evaluation
- `llm.25_audio_spectrogram_tokens` — Waveforms, STFT, and Audio Patches
- `llm.26_audio_temporal_attention` — Variable-Duration Audio and Attention
- `llm.27_tiny_audio_language_model` — Audio-Text Generation and Evaluation

Planned:

No additional language-model lesson is currently scheduled. Future multimodal
extensions should carry forward existing SFT, generation, and evaluation code
instead of asking learners to reimplement those mechanics for every modality.

### World models

Completed:

- `wm.01_vae` — Variational Autoencoder
- `wm.02_latent_dynamics` — Latent Dynamics
- `wm.03_mdn_rnn` — MDN-RNN
- `wm.04_world_model_loop` — World Model Loop
- `wm.05_cem_planning` — CEM Planning

Planned:

- `wm.06_jepa_latent_prediction` — Predict Targets in Representation Space
- `wm.07_action_conditioned_jepa` — Actions and Predictive Representations
- `wm.08_imagined_rollouts` — Latent Imagination and Lambda Returns
- `wm.09_actor_critic_from_imagination` — Actor and Value Learning in Dreams
- `wm.10_tiny_vla_policy` — Vision, Language, and Action Tokens
- `wm.11_joint_world_action_model` — Predict Futures and Actions Together
- `wm.12_wam_imagine_then_act` — Receding-Horizon Planning with a WAM
- `wm.13_stochastic_world_action_model` — Multiple Futures and Action Strategies
- `wm.14_uncertainty_aware_planning` — Model Disagreement and Safer Plans

The lessons should form one controlled comparison using tiny synthetic
trajectories, moving shapes, language goals, and discrete or two-dimensional
continuous actions. No lesson should require a robot simulator, downloaded
dataset, pretrained foundation model, or long training run.

The intended scope boundaries are:

- `wm.06` implements masked latent prediction, an online/target encoder,
  stop-gradient, an exponential-moving-average update, and latent-space loss. It
  compares JEPA prediction with VAE pixel reconstruction without reproducing a
  large vision architecture.
- `wm.07` adds one-step action-conditioned latent prediction and short goal-based
  rollouts. The completed representation learner from `wm.06` is provided.
- `wm.08` implements imagined latent rollouts and lambda returns only. Reward and
  continuation predictors are small provided modules.
- `wm.09` adds actor and value losses to the provided imagination code. Splitting
  these lessons avoids placing an entire Dreamer-style system in one scaffold.
- `wm.10` reuses provided visual, video, and language encoders and focuses on
  action representation, behavior-cloning loss, and short action chunks. It is
  explicitly a reactive policy baseline, not an explicit world model.
- `wm.11` uses a shared latent representation with a next-latent objective and an
  action objective. It compares reactive VLA prediction with joint world/action
  prediction but does not generate pixels or use a diffusion action decoder.
- `wm.12` carries forward the trained joint WAM and existing CEM utilities. The
  learner rolls out candidate action chunks, scores predicted goal progress,
  executes the first action, and replans; optimizer mechanics are provided.
- `wm.13` carries forward the WAM and MDN utilities. The learner represents,
  samples, and scores several coherent future/action hypotheses instead of
  averaging incompatible strategies.
- `wm.14` distinguishes model disagreement from valid multimodal futures. A small
  ensemble supplies epistemic uncertainty for risk-penalized action selection.

WAM lessons should remain latent-space exercises with action chunks of roughly
three to five steps. Video generation, photorealistic prediction, cross-embodiment
robotics, and real-time deployment belong in later extensions rather than these
foundational lessons.

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

- consolidate comparisons across native image, video, and audio lessons

### World models

- begin JEPA-style representation learning with `wm.06`
- connect actions to predictive representations with `wm.07`

### Diffusion models

- continue with `diffusion.10_consistency_models`

### Stage 2: deepen track coverage

- build imagined rollouts and actor-critic learning with `wm.08` through `wm.09`
- add the reactive VLA baseline with `wm.10`
- diffusion noise prediction, DDPM sampling, and tiny U-Net denoising

### Stage 3: connect to broader research patterns

- compare inference and modality behavior through richer reports
- connect future prediction and action generation with `wm.11` through `wm.13`
- add uncertainty-aware WAM planning with `wm.14`
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
