from datetime import UTC, datetime
from pathlib import Path

from rgym.report import generate_report


def test_generate_report_includes_metadata_files_and_review(tmp_path: Path) -> None:
    (tmp_path / "lesson.yaml").write_text(
        "\n".join(
            [
                "id: example.lesson",
                "title: Example Lesson",
                "test_command: uv run pytest tests",
                "run_command: uv run python scripts/run_demo.py",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "review.md").write_text(
        "# Questions\n\n- What did you learn?\n", encoding="utf-8"
    )

    report_path = generate_report(
        tmp_path,
        generated_at=datetime(2026, 6, 21, 12, 0, tzinfo=UTC),
    )
    report = report_path.read_text(encoding="utf-8")

    assert "# ResearchGym Report: Example Lesson" in report
    assert "2026-06-21T12:00:00+00:00" in report
    assert "`implementation.py`" in report
    assert "`uv run pytest tests`" in report
    assert "What did you learn?" in report
