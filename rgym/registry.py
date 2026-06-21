"""Discover and load ResearchGym lessons."""

from pathlib import Path
from typing import Any

import yaml

from rgym.lesson import Lesson

REQUIRED_FIELDS = (
    "id",
    "title",
    "track",
    "level",
    "summary",
    "entrypoint",
    "test_command",
    "run_command",
)


class LessonRegistryError(ValueError):
    """Raised when lesson metadata is missing or invalid."""


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LessonRegistryError(f"Could not read {metadata_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise LessonRegistryError(f"{metadata_path} must contain a YAML mapping")

    missing = [field for field in REQUIRED_FIELDS if not data.get(field)]
    if missing:
        fields = ", ".join(missing)
        raise LessonRegistryError(
            f"{metadata_path} is missing required fields: {fields}"
        )
    return data


def _load_lesson(metadata_path: Path) -> Lesson:
    data = _load_metadata(metadata_path)
    return Lesson(
        id=str(data["id"]),
        title=str(data["title"]),
        track=str(data["track"]),
        level=str(data["level"]),
        summary=str(data["summary"]),
        path=metadata_path.parent.resolve(),
        entrypoint=str(data["entrypoint"]),
        test_command=str(data["test_command"]),
        run_command=str(data["run_command"]),
    )


def discover_lessons(root: Path) -> list[Lesson]:
    """Return all lessons below ``root/tracks`` sorted by lesson id."""

    tracks_path = root.resolve() / "tracks"
    if not tracks_path.is_dir():
        return []

    lessons = [
        _load_lesson(metadata_path)
        for metadata_path in tracks_path.glob("**/lesson.yaml")
    ]

    seen: set[str] = set()
    for lesson in lessons:
        if lesson.id in seen:
            raise LessonRegistryError(f"Duplicate lesson id: {lesson.id}")
        seen.add(lesson.id)

    return sorted(lessons, key=lambda lesson: lesson.id)


def get_lesson(root: Path, lesson_id: str) -> Lesson:
    """Return one lesson by id or raise ``LessonRegistryError``."""

    for lesson in discover_lessons(root):
        if lesson.id == lesson_id:
            return lesson
    raise LessonRegistryError(f"Unknown lesson: {lesson_id}")
