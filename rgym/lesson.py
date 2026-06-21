"""Lesson metadata types."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Lesson:
    """Metadata loaded from a lesson.yaml file."""

    id: str
    title: str
    track: str
    level: str
    summary: str
    path: Path
    entrypoint: str
    test_command: str
    run_command: str
