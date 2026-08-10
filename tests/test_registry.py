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
        "wm.01_vae",
        "wm.02_latent_dynamics",
        "wm.03_mdn_rnn",
        "wm.04_world_model_loop",
        "wm.05_cem_planning",
    ]
    assert lessons[0].title == "Noise Schedules and the Forward Process"
    assert lessons[1].title == "Epsilon Prediction Objective"
    assert lessons[2].title == "Reverse DDPM Sampling"
    assert lessons[3].title == "Tiny U-Net Denoiser"
    assert lessons[4].title == "Deterministic DDIM Sampling"
    assert lessons[5].title == "Classifier-Free Guidance"
    assert lessons[6].title == "Latent Diffusion"
    assert lessons[7].title == "Cross-Attention Conditioning"
    assert lessons[8].title == "Flow Matching and ODE Sampling"
    assert lessons[9].title == "Bigram Language Model"
    assert lessons[10].title == "Tokenizer Fundamentals"
    assert lessons[11].title == "Causal Self-Attention"
    assert lessons[12].title == "Transformer Block"
    assert lessons[13].title == "Tiny GPT"
    assert lessons[14].title == "KV Cache"
    assert lessons[15].title == "Direct Preference Optimization"
    assert lessons[16].title == "Supervised Fine-Tuning Data"
    assert lessons[17].title == "Low-Rank Adaptation"
    assert lessons[18].title == "Language Model Sampling"
    assert lessons[19].title == "Images as Patch Tokens"
    assert lessons[20].title == "Visual Transformer Blocks"
    assert lessons[21].title == "Unified Image and Text Tokens"
    assert lessons[22].title == "Variational Autoencoder"
    assert lessons[23].title == "Latent Dynamics"
    assert lessons[24].title == "MDN-RNN"
    assert lessons[25].title == "World Model Loop"
    assert lessons[26].title == "CEM Planning"


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
