from pathlib import Path

import pytest

from rgym.registry import LessonRegistryError, discover_lessons, get_lesson


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_discover_lessons_returns_sorted_mvp_lessons() -> None:
    lessons = discover_lessons(PROJECT_ROOT)

    assert [lesson.id for lesson in lessons] == [
        "diffusion.01_forward_process",
        "diffusion.02_noise_prediction",
        "diffusion.03_ddpm_sampling",
        "diffusion.04_tiny_unet",
        "diffusion.05_ddim_sampling",
        "diffusion.06_classifier_free_guidance",
        "diffusion.07_latent_diffusion",
        "diffusion.08_cross_attention_conditioning",
        "diffusion.09_flow_matching",
        "diffusion.10_prediction_parameterizations",
        "diffusion.11_tiny_diffusion_transformer",
        "diffusion.12_masked_discrete_diffusion",
        "diffusion.13_tiny_diffusion_language_model",
        "diffusion.14_spatiotemporal_denoiser",
        "llm.01_bigram_lm",
        "llm.02_tokenizer",
        "llm.03_causal_attention",
        "llm.04_transformer_block",
        "llm.05_tiny_gpt",
        "llm.06_kv_cache",
        "llm.07_dpo",
        "llm.08_sft_data",
        "llm.09_lora",
        "llm.10_sampling",
        "llm.11_vision_patch_embeddings",
        "llm.12_vision_attention",
        "llm.13_multimodal_sequence",
        "llm.14_multimodal_attention_mask",
        "llm.15_tiny_native_vlm",
        "llm.16_multimodal_sft_data",
        "llm.17_multimodal_generation",
        "llm.18_multimodal_eval",
        "llm.19_quantization",
        "llm.20_speculative_decoding",
        "llm.21_paged_kv_and_continuous_batching",
        "llm.22_video_tubelet_embeddings",
        "llm.23_factorized_video_attention",
        "llm.24_tiny_video_language_model",
        "llm.25_audio_spectrogram_tokens",
        "llm.26_audio_temporal_attention",
        "llm.27_tiny_audio_language_model",
        "llm.28_rotary_position_embeddings",
        "llm.29_grouped_query_attention",
        "llm.30_gated_linear_attention",
        "llm.31_rl_rollouts_and_advantages",
        "llm.32_ppo",
        "llm.33_grpo",
        "wm.01_vae",
        "wm.02_latent_dynamics",
        "wm.03_mdn_rnn",
        "wm.04_world_model_loop",
        "wm.05_cem_planning",
        "wm.06_jepa_latent_prediction",
        "wm.07_action_conditioned_jepa",
        "wm.08_imagined_rollouts",
        "wm.09_actor_critic_from_imagination",
        "wm.10_tiny_vla_policy",
        "wm.11_joint_world_action_model",
        "wm.12_wam_imagine_then_act",
        "wm.13_stochastic_world_action_model",
        "wm.14_uncertainty_aware_planning",
    ]
    assert [lesson.title for lesson in lessons] == [
        "Noise Schedules and the Forward Process",
        "Epsilon Prediction Objective",
        "Reverse DDPM Sampling",
        "Tiny U-Net Denoiser",
        "Deterministic DDIM Sampling",
        "Classifier-Free Guidance",
        "Latent Diffusion",
        "Cross-Attention Conditioning",
        "Flow Matching and ODE Sampling",
        "Epsilon, Clean-Sample, and Velocity Prediction",
        "Tiny Diffusion Transformer",
        "Absorbing-Mask Corruption for Tokens",
        "Tiny Diffusion Language Model",
        "Spatiotemporal Video Denoiser",
        "Bigram Language Model",
        "Tokenizer Fundamentals",
        "Causal Self-Attention",
        "Transformer Block",
        "Tiny GPT",
        "KV Cache",
        "Direct Preference Optimization",
        "Supervised Fine-Tuning Data",
        "Low-Rank Adaptation",
        "Language Model Sampling",
        "Images as Patch Tokens",
        "Visual Transformer Blocks",
        "Unified Image and Text Tokens",
        "Visual Prefix and Causal Text",
        "Tiny Native Vision-Language Model",
        "Image-Text Conversations",
        "Image Prefill, KV Cache, and Decoding",
        "Tiny Vision-Language Evaluation Harness",
        "Weight Quantization",
        "Draft, Verify, and Correct",
        "KV Blocks and Request Scheduling",
        "Videos as Spatiotemporal Tokens",
        "Spatial and Temporal Attention",
        "Video-Text Generation and Evaluation",
        "Waveforms, STFT, and Audio Patches",
        "Variable-Duration Audio and Attention",
        "Audio-Text Generation and Evaluation",
        "Rotary Position Embeddings and Cache Offsets",
        "Shared Key-Value Heads and Smaller Caches",
        "Gated Linear Attention and Selective Memory",
        "Token Rewards and Advantage Estimation",
        "Clipped Policy and Value Updates",
        "Group-Relative Policy Optimization",
        "Variational Autoencoder",
        "Latent Dynamics",
        "MDN-RNN",
        "World Model Loop",
        "CEM Planning",
        "Predict Targets in Representation Space",
        "Actions and Predictive Representations",
        "Latent Imagination and Lambda Returns",
        "Actor and Value Learning in Dreams",
        "Vision, Language, and Action Chunks",
        "Predict Futures and Actions Together",
        "Receding-Horizon Planning with a WAM",
        "Multiple Futures and Action Strategies",
        "Model Disagreement and Safer Plans",
    ]


def test_get_lesson_returns_requested_lesson() -> None:
    lesson = get_lesson(PROJECT_ROOT, "wm.02_latent_dynamics")

    assert lesson.track == "world_models"
    assert lesson.entrypoint == "implementation.py"
    assert lesson.path.name == "02_latent_dynamics"


def test_get_lesson_rejects_unknown_id() -> None:
    with pytest.raises(LessonRegistryError, match="Unknown lesson"):
        get_lesson(PROJECT_ROOT, "missing.lesson")


def test_discover_lessons_rejects_missing_fields(tmp_path: Path) -> None:
    lesson_path = tmp_path / "tracks" / "example"
    lesson_path.mkdir(parents=True)
    (lesson_path / "lesson.yaml").write_text("id: incomplete\n", encoding="utf-8")

    with pytest.raises(LessonRegistryError, match="missing required fields"):
        discover_lessons(tmp_path)


def test_discover_lessons_rejects_duplicate_ids(tmp_path: Path) -> None:
    metadata = """
id: duplicate.lesson
title: Duplicate
track: example
level: fundamental
summary: Duplicate metadata.
entrypoint: implementation.py
test_command: uv run pytest tests
run_command: uv run python scripts/run_demo.py
""".strip()
    for directory in ("one", "two"):
        lesson_path = tmp_path / "tracks" / directory
        lesson_path.mkdir(parents=True)
        (lesson_path / "lesson.yaml").write_text(metadata, encoding="utf-8")

    with pytest.raises(LessonRegistryError, match="Duplicate lesson id"):
        discover_lessons(tmp_path)
