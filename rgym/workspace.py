"""Create editable lesson workspaces."""

import shutil
from pathlib import Path

from rgym.lesson import Lesson


class WorkspaceError(ValueError):
    """Raised when a lesson workspace cannot be created."""


def create_workspace(
    lesson: Lesson,
    workspace_root: Path,
    *,
    force: bool = False,
) -> Path:
    """Copy ``lesson`` into an editable directory below ``workspace_root``."""

    destination = workspace_root / lesson.id
    if destination.exists():
        if not force:
            raise WorkspaceError(
                f"Workspace already exists: {destination}\nUse --force to recreate it."
            )
        shutil.rmtree(destination)

    destination.mkdir(parents=True)
    for source in lesson.path.iterdir():
        if source.name == "solution.py":
            continue

        target_name = (
            "implementation.py" if source.name == "scaffold.py" else source.name
        )
        target = destination / target_name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    implementation = destination / lesson.entrypoint
    if not implementation.is_file():
        shutil.rmtree(destination)
        raise WorkspaceError(f"Lesson is missing scaffold.py: {lesson.path}")

    return destination
