"""ResearchGym command-line interface."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rgym.registry import LessonRegistryError, discover_lessons, get_lesson

app = typer.Typer(
    name="rgym",
    help="Learn foundational ML ideas by implementing them from scratch.",
    no_args_is_help=True,
)
console = Console()


def project_root() -> Path:
    """Return the repository root containing the installed package."""

    return Path(__file__).resolve().parent.parent


@app.command("list")
def list_lessons() -> None:
    """List available lessons."""

    try:
        lessons = discover_lessons(project_root())
    except LessonRegistryError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not lessons:
        console.print("No lessons found.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Lesson ID")
    table.add_column("Title")
    table.add_column("Track")
    table.add_column("Level")
    for lesson in lessons:
        table.add_row(lesson.id, lesson.title, lesson.track, lesson.level)
    console.print(table)


@app.command("inspect")
def inspect_lesson(
    lesson_id: Annotated[str, typer.Argument(help="Lesson id to inspect.")],
) -> None:
    """Show metadata and available documentation for one lesson."""

    try:
        lesson = get_lesson(project_root(), lesson_id)
    except LessonRegistryError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    docs = sorted(path.name for path in lesson.path.glob("*.md"))
    console.print(f"[bold]{lesson.title}[/bold]")
    console.print(f"ID: {lesson.id}")
    console.print(f"Track: {lesson.track}")
    console.print(f"Level: {lesson.level}")
    console.print(f"Summary: {lesson.summary}")
    console.print(f"Path: {lesson.path}")
    console.print(f"Entrypoint: {lesson.entrypoint}")
    console.print(f"Test command: {lesson.test_command}")
    console.print(f"Run command: {lesson.run_command}")
    console.print(f"Available docs: {', '.join(docs) if docs else 'none'}")


if __name__ == "__main__":
    app()
