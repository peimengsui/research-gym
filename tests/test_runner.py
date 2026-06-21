import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rgym.runner import RunnerError, read_hint, run_workspace_command


def write_metadata(workspace: Path) -> None:
    (workspace / "lesson.yaml").write_text(
        "\n".join(
            [
                "id: example.lesson",
                "test_command: uv run pytest tests",
                "run_command: uv run python scripts/run_demo.py",
            ]
        ),
        encoding="utf-8",
    )


def test_run_workspace_command_uses_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_metadata(tmp_path)
    calls: list[tuple[list[str], Path, bool, dict[str, str]]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> SimpleNamespace:
        calls.append((command, cwd, check, env))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("rgym.runner.subprocess.run", fake_run)

    assert run_workspace_command(tmp_path, "test_command") == 7
    command, cwd, check, env = calls[0]
    assert command == ["uv", "run", "pytest", "tests"]
    assert cwd == tmp_path
    assert check is False
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path.resolve())


def test_run_workspace_command_requires_workspace(tmp_path: Path) -> None:
    with pytest.raises(RunnerError, match="No lesson.yaml"):
        run_workspace_command(tmp_path, "test_command")


def test_read_hint_returns_requested_section(tmp_path: Path) -> None:
    (tmp_path / "hints.md").write_text(
        "# Hints\n\n## Hint 1\n\nFirst.\n\n## Hint 2\n\nSecond.\n",
        encoding="utf-8",
    )

    assert read_hint(tmp_path, 2) == "## Hint 2\n\nSecond."


def test_read_hint_rejects_unavailable_level(tmp_path: Path) -> None:
    (tmp_path / "hints.md").write_text("## Hint 1\n\nFirst.\n", encoding="utf-8")

    with pytest.raises(RunnerError, match="choose 1-1"):
        read_hint(tmp_path, 2)
