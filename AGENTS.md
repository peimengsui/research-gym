# AGENTS.md

## Project

This repository is `research-gym`, an open-source learning environment for re-implementing foundational LLM and world-model ideas from scratch.

The project is not a normal model library. It is a guided research-implementation gym.

Each lesson should help a learner move through this loop:

```text
read concept
→ inspect scaffold
→ fill implementation
→ run tests
→ run demo
→ review code
→ compare with reference
→ generate report
```

## Core philosophy

Do not hide the implementation from the learner.

Prefer:

* Python source files
* Markdown guides
* small tests
* tiny reproducible demos
* clear scaffolds
* explicit tensor shapes
* readable reference implementations

Avoid:

* Jupyter notebooks
* large training jobs
* complex distributed systems
* unnecessary abstractions
* premature web UI
* heavy dependencies
* magic code generation that bypasses learning

## Week 1–2 MVP goal

Build the first usable CLI-first version of ResearchGym.

The MVP should support:

* listing lessons
* inspecting a lesson
* starting a lesson into a workspace
* running tests
* running a demo script
* showing static hints
* generating a simple Markdown report

The MVP should include two lessons:

1. `llm.01_bigram_lm`
2. `wm.01_vae`

The first version should be CPU-friendly. Do not require GPU for tests.

## Preferred stack

Use:

* Python 3.11+
* uv for Python version, virtual environment, and dependency management
* PyTorch
* pytest
* Typer for CLI
* PyYAML for lesson metadata
* Rich optional, only if useful
* ruff for formatting/linting if easy

Do not use:

* Jupyter notebooks
* Hydra
* Lightning
* Weights & Biases
* FastAPI
* Next.js
* Docker
* distributed training

These can be added later.

## Repository layout

Target layout:

```text
research-gym/
  AGENTS.md
  PLANS.md
  README.md
  pyproject.toml
  uv.lock

  rgym/
    __init__.py
    cli.py
    lesson.py
    registry.py
    workspace.py
    runner.py
    report.py

  tracks/
    llm/
      01_bigram_lm/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_bigram.py
        scripts/
          run_demo.py

      02_tokenizer/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_tokenizer.py
        scripts/
          run_demo.py

    world_models/
      01_vae/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_vae.py
        scripts/
          run_demo.py

  tests/
    test_registry.py
    test_cli_smoke.py
```

## CLI behavior

Implement command name:

```bash
rgym
```

Required commands:

```bash
rgym list
rgym inspect <lesson_id>
rgym start <lesson_id>
rgym test
rgym run
rgym hint
rgym report
```

Optional if easy:

```bash
rgym doctor
rgym reset
```

## Workspace behavior

`rgym start <lesson_id>` should copy a lesson into:

```text
workspace/<lesson_id>/
```

The learner should edit:

```text
workspace/<lesson_id>/implementation.py
```

The source lesson should remain unchanged.

When copying:

* `scaffold.py` becomes `implementation.py`
* tests are copied
* scripts are copied
* Markdown files are copied
* `solution.py` may be copied as `solution.locked.py` or left in source only for MVP

For MVP, it is okay if `solution.py` remains visible in the source tree. Do not over-engineer solution hiding.

## Lesson metadata

Each lesson should have `lesson.yaml`:

```yaml
id: llm.01_bigram_lm
title: Bigram Language Model
track: llm
level: fundamental
summary: Build the smallest useful next-token language model.
entrypoint: implementation.py
test_command: uv run pytest tests
run_command: uv run python scripts/run_demo.py
```

## Testing expectations

Every Codex implementation step should end by running:

```bash
uv run pytest
```

If the CLI is implemented, also run:

```bash
uv run rgym list
uv run rgym inspect llm.01_bigram_lm
uv run rgym start llm.01_bigram_lm
cd workspace/llm.01_bigram_lm
uv run rgym test
uv run rgym run
```

If there is a failure, fix the implementation or update the tests only if the test is genuinely wrong.

## Code style

Prefer simple readable code.

Use type hints where helpful.

Avoid clever framework abstractions.

Keep lesson code easy for a beginner/intermediate ML engineer to read.

## Dependency policy

Use `uv` for all dependency and environment management.

* Declare dependencies in `pyproject.toml`.
* Commit `uv.lock`.
* Use `uv sync` to create or update the local environment.
* Use `uv add <package>` for runtime dependencies.
* Use `uv add --dev <package>` for development dependencies.
* Use `uv run <command>` for Python, tests, and CLI commands.
* Do not use `pip install`, `python -m venv`, Poetry, Pipenv, or Conda for the project workflow.

Ask before adding heavy dependencies.

Allowed dependencies for MVP:

* torch
* pytest
* typer
* pyyaml
* rich

Do not add dependencies for UI, web backend, experiment tracking, or distributed training.

## ResearchGym-specific rules

When writing scaffolds:

* Include TODO comments.
* Include expected input and output shapes.
* Include conceptual hints.
* Do not include the full implementation in `scaffold.py`.

When writing solutions:

* Keep the implementation minimal and readable.
* Prefer explicit tensor shapes.
* Avoid optimization tricks in the first version.

When writing tests:

* Test shape correctness.
* Test basic behavior.
* Test gradient flow where relevant.
* Keep tests CPU-friendly and fast.

When writing guides:

* Explain the research idea.
* Explain the implementation steps.
* Explain common bugs.
* Include commands to run tests and demos.

## Done means

A task is not done until:

* files are created in the expected layout
* `uv run pytest` passes
* CLI smoke tests pass
* README explains how to install and run
* both MVP lessons can be started, tested, and run
* generated workspaces do not modify source lessons
