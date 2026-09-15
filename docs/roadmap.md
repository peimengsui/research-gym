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
- complete Rotary Position Embeddings and Cache Offsets lesson
- complete Shared Key-Value Heads and Smaller Caches lesson
- complete Gated Linear Attention and Selective Memory lesson
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
- `llm.28_rotary_position_embeddings` — Rotary Position Embeddings and Cache Offsets
- `llm.29_grouped_query_attention` — Shared Key-Value Heads and Smaller Caches
- `llm.30_gated_linear_attention` — Gated Linear Attention and Selective Memory

Planned:

- `llm.31_rl_rollouts_and_advantages` — Token Rewards and Advantage Estimation
- `llm.32_ppo` — Clipped Policy and Value Updates
- `llm.33_grpo` — Group-Relative Policy Optimization

The attention lessons should form a controlled progression over the existing
causal-attention and KV-cache code:

- `llm.28` rotates query and key pairs, preserves vector norms, and handles
  nonzero position offsets during cached decoding.
- `llm.29` gives query heads fewer shared key/value heads, checks equivalence to
  multi-head attention in the degenerate case, and measures KV-cache reduction.
- `llm.30` implements the practical key-gated GLA recurrence with low-rank,
  data-dependent forget gates, fixed-size matrix memory, an equivalent explicit
  parallel oracle, per-head normalization, and output gating. It distinguishes
  those learning mechanics from the production chunkwise GPU kernels.

The reinforcement-learning lessons should use tiny generated responses and a
provided deterministic reward so the exercises stay reproducible and do not
require a learned reward model:

- `llm.31` collects response-only masks, old-policy and reference log
  probabilities, token-level KL-shaped rewards, returns, and advantages.
- `llm.32` carries those frozen rollouts forward and focuses on probability
  ratios, clipping, value regression, entropy, and repeated minibatch updates.
- `llm.33` samples several responses per prompt and replaces the learned value
  baseline with normalized group-relative rewards for a direct PPO comparison.

An explicit preference-trained reward-model lesson remains an optional later
extension for completing the classic RLHF pipeline. Future modality lessons
should continue carrying forward existing SFT, generation, and evaluation code
instead of asking learners to reimplement those mechanics for every modality.

### World models

Completed:

- `wm.01_vae` — Variational Autoencoder
- `wm.02_latent_dynamics` — Latent Dynamics
- `wm.03_mdn_rnn` — MDN-RNN
- `wm.04_world_model_loop` — World Model Loop
- `wm.05_cem_planning` — CEM Planning
- `wm.06_jepa_latent_prediction` — Predict Targets in Representation Space
- `wm.07_action_conditioned_jepa` — Actions and Predictive Representations
- `wm.08_imagined_rollouts` — Latent Imagination and Lambda Returns
- `wm.09_actor_critic_from_imagination` — Actor and Value Learning in Dreams
- `wm.10_tiny_vla_policy` — Vision, Language, and Action Chunks
- `wm.11_joint_world_action_model` — Predict Futures and Actions Together
- `wm.12_wam_imagine_then_act` — Receding-Horizon Planning with a WAM
- `wm.13_stochastic_world_action_model` — Multiple Futures and Action Strategies
- `wm.14_uncertainty_aware_planning` — Model Disagreement and Safer Plans

All currently planned world-model lessons are implemented.

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

Stage 1 — modern denoising foundations:

- `diffusion.10_prediction_parameterizations` — Epsilon, Clean-Sample, and Velocity Prediction
- `diffusion.11_tiny_diffusion_transformer` — Denoising Latent Patches with a Transformer

Stage 2 — discrete text diffusion:

- `diffusion.12_masked_discrete_diffusion` — Absorbing-Mask Corruption for Tokens
- `diffusion.13_tiny_diffusion_language_model` — Generate Text by Iterative Unmasking

Stage 3 — video diffusion:

- `diffusion.14_spatiotemporal_denoiser` — Couple Spatial and Temporal Denoising
- `diffusion.15_tiny_video_diffusion` — Generate Tiny Moving-Dot Videos
- `diffusion.16_conditioned_video_diffusion` — First-Frame and Text-Guided Video

Stage 4 — fast generation:

- `diffusion.17_consistency_models` — One/Few-Step Consistency Models

The intended scope boundaries are:

- `diffusion.10` converts among epsilon, clean-sample, and velocity targets,
  derives signal-to-noise ratio by timestep, and compares weighted losses. It
  should remain a tensor exercise rather than another image model.
- `diffusion.11` replaces the U-Net with a tiny transformer over latent patches.
  It focuses on patchification, timestep conditioning, transformer blocks, and
  unpatchification without reproducing a production-scale DiT.
- `diffusion.12` is the discrete counterpart of the forward-process lesson. It
  implements an absorbing mask token, a masking schedule, direct corruption,
  and loss masks while leaving the denoising network for the next lesson.
- `diffusion.13` carries the discrete process forward into a bidirectional token
  denoiser and iterative confidence-based unmasking. Padding, special tokens,
  and already-fixed tokens must remain unchanged.
- `diffusion.14` introduces `[batch, channels, frames, height, width]` tensors,
  reuses provided spatial blocks, and adds temporal mixing. It tests temporal
  coupling without training a complete video generator.
- `diffusion.15` carries that denoiser into a DDPM or DDIM loop over synthetic
  four-frame moving-dot clips and uses small motion-continuity checks instead of
  heavyweight perceptual metrics.
- `diffusion.16` adds a preserved first-frame condition and tiny text motion
  instructions, reusing existing cross-attention and classifier-free-guidance
  mechanics instead of asking learners to implement them again.
- `diffusion.17` returns to sampling efficiency after the image, text, and video
  mechanics are established. It remains a tiny one/few-step consistency
  exercise rather than a large distillation pipeline.

Text diffusion here means generating discrete text tokens. Text-conditioned
visual generation is already introduced by `diffusion.06` and `diffusion.08`.
The video lessons should stay in pixel space or use a provided deterministic
compressor; learned video autoencoders, cascaded super-resolution, FID/FVD,
downloaded video datasets, and long training runs remain out of scope.

## Implementation stages

### Stage 1: continue core tracks

### Language models

- consolidate comparisons across RoPE, grouped-query, and gated linear attention
- consolidate comparisons across native image, video, and audio lessons

### World models

- consolidate the progression from representation learning in `wm.06` through
  action-conditioned latent imagination in `wm.08`

### Diffusion models

- add prediction parameterizations and a tiny DiT with `diffusion.10` and
  `diffusion.11`

### Stage 2: deepen track coverage

- connect rollout bookkeeping, PPO, and GRPO with `llm.31` through `llm.33`
- consolidate the completed latent-imagination actor-critic sequence through
  `wm.09`
- consolidate the reactive VLA baseline in `wm.10`
- add joint action and next-latent prediction with `wm.11`
- add receding-horizon WAM planning with `wm.12`
- introduce discrete text diffusion with `diffusion.12` and `diffusion.13`

### Stage 3: connect to broader research patterns

- compare inference and modality behavior through richer reports
- connect future prediction and action generation with `wm.11` through `wm.13`
- add uncertainty-aware WAM planning with `wm.14`
- extend diffusion through video tensors and conditioning with `diffusion.14`
  through `diffusion.16`
- revisit fast sampling with consistency models in `diffusion.17`
- lesson authoring validation and contribution templates
- richer reports that summarize tests, demos, and learner reflection

## Guiding constraints

New lessons should remain:

- understandable from source
- runnable on CPU for tests
- small enough to complete incrementally
- explicit about tensor shapes and objectives
- independent of notebooks, distributed training, and hosted experiment tools
