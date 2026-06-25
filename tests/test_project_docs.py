from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_project_documentation_exists() -> None:
    required_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "PLANS.md",
        PROJECT_ROOT / "docs" / "design.md",
        PROJECT_ROOT / "docs" / "roadmap.md",
    ]

    assert all(path.is_file() for path in required_files)


def test_readme_contains_complete_workspace_workflow() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    commands = [
        "uv sync --locked",
        "uv run rgym list",
        "uv run rgym inspect llm.01_bigram_lm",
        "uv run rgym inspect llm.02_tokenizer",
        "uv run rgym inspect wm.01_vae",
        "uv run rgym inspect wm.02_latent_dynamics",
        "uv run rgym start llm.01_bigram_lm",
        "uv run rgym start llm.02_tokenizer",
        "uv run rgym start wm.01_vae",
        "uv run rgym start wm.02_latent_dynamics",
        "uv run rgym test",
        "uv run rgym run",
        "uv run rgym hint",
        "uv run rgym report",
    ]

    assert all(command in readme for command in commands)


def test_repository_contains_no_notebooks() -> None:
    notebooks = [
        path
        for path in PROJECT_ROOT.rglob("*.ipynb")
        if ".venv" not in path.parts and "workspace" not in path.parts
    ]

    assert notebooks == []


def test_mvp_lessons_have_complete_structure() -> None:
    lesson_paths = [
        PROJECT_ROOT / "tracks" / "llm" / "01_bigram_lm",
        PROJECT_ROOT / "tracks" / "llm" / "02_tokenizer",
        PROJECT_ROOT / "tracks" / "world_models" / "01_vae",
        PROJECT_ROOT / "tracks" / "world_models" / "02_latent_dynamics",
    ]
    required_paths = [
        "lesson.yaml",
        "README.md",
        "concept.md",
        "guide.md",
        "scaffold.py",
        "solution.py",
        "hints.md",
        "review.md",
        "scripts/run_demo.py",
    ]

    for lesson_path in lesson_paths:
        assert all((lesson_path / path).is_file() for path in required_paths)
        assert any((lesson_path / "tests").glob("test_*.py"))
