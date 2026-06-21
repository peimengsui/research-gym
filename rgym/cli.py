"""ResearchGym command-line interface."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rgym.registry import LessonRegistryError, discover_lessons, get_lesson
from rgym.report import generate_report
from rgym.runner import RunnerError, read_hint, run_workspace_command
from rgym.workspace import WorkspaceError, create_workspace

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


@app.command("start")
def start_lesson(
    lesson_id: Annotated[str, typer.Argument(help="Lesson id to start.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace an existing workspace."),
    ] = False,
) -> None:
    """Create an editable workspace for a lesson."""

    try:
        lesson = get_lesson(project_root(), lesson_id)
        destination = create_workspace(
            lesson,
            project_root() / "workspace",
            force=force,
        )
    except (LessonRegistryError, WorkspaceError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Workspace created: {destination}")
    console.print(f"Edit: {destination / lesson.entrypoint}")


def _run_command(command_field: str) -> None:
    try:
        exit_code = run_workspace_command(Path.cwd(), command_field)
    except RunnerError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if exit_code:
        raise typer.Exit(code=exit_code)


@app.command("test")
def test_lesson() -> None:
    """Run the current workspace's tests."""

    _run_command("test_command")


@app.command("run")
def run_lesson() -> None:
    """Run the current workspace's demo."""

    _run_command("run_command")


@app.command("hint")
def show_hint(
    level: Annotated[
        int,
        typer.Option("--level", min=1, help="Hint number to display."),
    ] = 1,
) -> None:
    """Show a static hint for the current workspace."""

    try:
        hint = read_hint(Path.cwd(), level)
    except RunnerError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(hint)


@app.command("report")
def report_lesson() -> None:
    """Generate a Markdown report for the current workspace."""

    try:
        report_path = generate_report(Path.cwd())
    except RunnerError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Report generated: {report_path}")


if __name__ == "__main__":
    app()
