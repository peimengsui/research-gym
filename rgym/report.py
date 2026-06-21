"""Generate a simple Markdown report for a lesson workspace."""

from datetime import UTC, datetime
from pathlib import Path

from rgym.runner import load_workspace_metadata


def generate_report(workspace: Path, *, generated_at: datetime | None = None) -> Path:
    """Write ``report.md`` in ``workspace`` and return its path."""

    metadata = load_workspace_metadata(workspace)
    timestamp = generated_at or datetime.now(UTC)
    files = sorted(
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and path.name != "report.md"
    )

    review_path = workspace / "review.md"
    reflection = (
        review_path.read_text(encoding="utf-8").strip()
        if review_path.is_file()
        else "No reflection questions are available."
    )

    content = "\n".join(
        [
            f"# ResearchGym Report: {metadata.get('title', 'Untitled lesson')}",
            "",
            f"- Lesson ID: `{metadata.get('id', 'unknown')}`",
            f"- Generated: `{timestamp.isoformat()}`",
            f"- Test command: `{metadata.get('test_command', 'not configured')}`",
            f"- Run command: `{metadata.get('run_command', 'not configured')}`",
            "",
            "## Files present",
            "",
            *[f"- `{file}`" for file in files],
            "",
            "## Reflection",
            "",
            reflection,
            "",
        ]
    )

    report_path = workspace / "report.md"
    report_path.write_text(content, encoding="utf-8")
    return report_path
