from pathlib import Path

import pytest

from rgym.registry import LessonRegistryError, discover_lessons, get_lesson


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_discover_lessons_returns_sorted_mvp_lessons() -> None:
    lessons = discover_lessons(PROJECT_ROOT)

    assert [lesson.id for lesson in lessons] == [
        "llm.01_bigram_lm",
        "llm.02_tokenizer",
        "llm.03_causal_attention",
        "llm.04_transformer_block",
        "wm.01_vae",
        "wm.02_latent_dynamics",
    ]
    assert lessons[0].title == "Bigram Language Model"
    assert lessons[1].title == "Tokenizer Fundamentals"
    assert lessons[2].title == "Causal Self-Attention"
    assert lessons[3].title == "Transformer Block"
    assert lessons[4].title == "Variational Autoencoder"
    assert lessons[5].title == "Latent Dynamics"


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
