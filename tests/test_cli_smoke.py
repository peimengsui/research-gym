from pathlib import Path

from typer.testing import CliRunner

from rgym.cli import app


runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Learn foundational ML ideas" in result.stdout


def test_list_shows_available_lessons() -> None:
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "llm.01_bigram_lm" in result.stdout
    assert "Bigram Language Model" in result.stdout
    assert "llm.02_tokenizer" in result.stdout
    assert "Tokenizer Fundamentals" in result.stdout
    assert "wm.01_vae" in result.stdout
    assert "Variational Autoencoder" in result.stdout


def test_inspect_shows_lesson_details() -> None:
    result = runner.invoke(app, ["inspect", "llm.01_bigram_lm"])

    assert result.exit_code == 0
    assert "Bigram Language Model" in result.stdout
    assert "Build the smallest useful next-token language model." in result.stdout
    assert "README.md" in result.stdout
    assert "uv run pytest tests" in result.stdout


def test_inspect_rejects_unknown_lesson() -> None:
    result = runner.invoke(app, ["inspect", "missing.lesson"])

    assert result.exit_code == 1
    assert "Unknown lesson: missing.lesson" in result.stdout


def test_start_creates_workspace(tmp_path: Path, monkeypatch) -> None:
    lesson_path = tmp_path / "tracks" / "example"
    lesson_path.mkdir(parents=True)
    (lesson_path / "lesson.yaml").write_text(
        "\n".join(
            [
                "id: example.lesson",
                "title: Example",
                "track: example",
                "level: fundamental",
                "summary: Example lesson.",
                "entrypoint: implementation.py",
                "test_command: uv run pytest tests",
                "run_command: uv run python scripts/run_demo.py",
            ]
        ),
        encoding="utf-8",
    )
    (lesson_path / "scaffold.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr("rgym.cli.project_root", lambda: tmp_path)

    result = runner.invoke(app, ["start", "example.lesson"])

    assert result.exit_code == 0
    assert (tmp_path / "workspace" / "example.lesson" / "implementation.py").is_file()


def test_hint_and_report_work_in_workspace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("lesson.yaml").write_text(
        "\n".join(
            [
                "id: example.lesson",
                "title: Example",
                "test_command: uv run pytest tests",
                "run_command: uv run python scripts/run_demo.py",
            ]
        ),
        encoding="utf-8",
    )
    Path("hints.md").write_text("## Hint 1\n\nKeep going.\n", encoding="utf-8")
    Path("implementation.py").write_text("VALUE = 1\n", encoding="utf-8")

    hint_result = runner.invoke(app, ["hint"])
    report_result = runner.invoke(app, ["report"])

    assert hint_result.exit_code == 0
    assert "Keep going." in hint_result.stdout
    assert report_result.exit_code == 0
    assert Path("report.md").is_file()
