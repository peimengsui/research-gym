from typer.testing import CliRunner

from rgym.cli import app


runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Learn foundational ML ideas" in result.stdout


def test_list_shows_both_lessons() -> None:
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "llm.01_bigram_lm" in result.stdout
    assert "Bigram Language Model" in result.stdout
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
