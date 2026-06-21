from pathlib import Path

import pytest

from rgym.lesson import Lesson
from rgym.workspace import WorkspaceError, create_workspace


def make_lesson(tmp_path: Path) -> Lesson:
    lesson_path = tmp_path / "source"
    lesson_path.mkdir()
    (lesson_path / "scaffold.py").write_text("VALUE = 1\n", encoding="utf-8")
    (lesson_path / "solution.py").write_text("VALUE = 2\n", encoding="utf-8")
    (lesson_path / "lesson.yaml").write_text("id: example.lesson\n", encoding="utf-8")
    tests_path = lesson_path / "tests"
    tests_path.mkdir()
    (tests_path / "test_example.py").write_text("def test_example(): pass\n")
    return Lesson(
        id="example.lesson",
        title="Example",
        track="example",
        level="fundamental",
        summary="Example lesson.",
        path=lesson_path,
        entrypoint="implementation.py",
        test_command="uv run pytest tests",
        run_command="uv run python scripts/run_demo.py",
    )


def test_create_workspace_copies_and_renames_scaffold(tmp_path: Path) -> None:
    lesson = make_lesson(tmp_path)

    destination = create_workspace(lesson, tmp_path / "workspace")

    assert (destination / "implementation.py").read_text() == "VALUE = 1\n"
    assert (destination / "lesson.yaml").is_file()
    assert (destination / "tests" / "test_example.py").is_file()
    assert not (destination / "scaffold.py").exists()
    assert not (destination / "solution.py").exists()
    assert (lesson.path / "scaffold.py").read_text() == "VALUE = 1\n"


def test_create_workspace_refuses_to_overwrite(tmp_path: Path) -> None:
    lesson = make_lesson(tmp_path)
    create_workspace(lesson, tmp_path / "workspace")

    with pytest.raises(WorkspaceError, match="Use --force"):
        create_workspace(lesson, tmp_path / "workspace")


def test_create_workspace_force_recreates_destination(tmp_path: Path) -> None:
    lesson = make_lesson(tmp_path)
    destination = create_workspace(lesson, tmp_path / "workspace")
    (destination / "learner-change.txt").write_text("remove me", encoding="utf-8")

    recreated = create_workspace(lesson, tmp_path / "workspace", force=True)

    assert recreated == destination
    assert not (recreated / "learner-change.txt").exists()
    assert (recreated / "implementation.py").is_file()
