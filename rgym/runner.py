"""Run commands and read hints from a lesson workspace."""

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


class RunnerError(ValueError):
    """Raised when the current directory is not a valid lesson workspace."""


def load_workspace_metadata(workspace: Path) -> dict[str, Any]:
    """Load metadata from ``workspace/lesson.yaml``."""

    metadata_path = workspace / "lesson.yaml"
    if not metadata_path.is_file():
        raise RunnerError(
            f"No lesson.yaml found in {workspace.resolve()}. "
            "Run this command from a lesson workspace."
        )

    try:
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RunnerError(f"Could not read {metadata_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RunnerError(f"{metadata_path} must contain a YAML mapping")
    return data


def run_workspace_command(workspace: Path, command_field: str) -> int:
    """Execute the command stored in ``command_field`` and return its exit code."""

    metadata = load_workspace_metadata(workspace)
    command = metadata.get(command_field)
    if not isinstance(command, str) or not command.strip():
        raise RunnerError(f"lesson.yaml is missing {command_field}")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(workspace.resolve())]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    result = subprocess.run(
        shlex.split(command),
        cwd=workspace,
        check=False,
        env=env,
    )
    return result.returncode


def read_hint(workspace: Path, level: int = 1) -> str:
    """Return one numbered section from ``hints.md``."""

    if level < 1:
        raise RunnerError("Hint level must be 1 or greater")

    hints_path = workspace / "hints.md"
    if not hints_path.is_file():
        raise RunnerError(f"No hints.md found in {workspace.resolve()}")

    sections: list[str] = []
    current: list[str] = []
    for line in hints_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current:
                sections.append("\n".join(current).strip())
            current = [line]
        elif current:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    if level > len(sections):
        raise RunnerError(
            f"Hint level {level} is unavailable; choose 1-{len(sections)}"
        )
    return sections[level - 1]
