# ResearchGym

ResearchGym is a CLI-first learning environment for re-implementing
foundational language-model and world-model ideas from scratch.

The project uses ordinary Python files, short guides, focused tests, and tiny
demos instead of notebooks. This keeps every implementation visible, easy to
review, and reproducible from the command line.

## Setup

ResearchGym requires Python 3.11 or newer and uses
[uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync
```

## Discover lessons

The Phase 0–1 CLI can list and inspect the two planned MVP lessons:

```bash
uv run rgym list
uv run rgym inspect llm.01_bigram_lm
uv run rgym inspect wm.01_vae
```

Run the test suite with:

```bash
uv run pytest
```

## Lesson workspace workflow

```bash
uv run rgym start llm.01_bigram_lm
cd workspace/llm.01_bigram_lm
uv run rgym test
uv run rgym run
uv run rgym hint
uv run rgym report
```

The Bigram Language Model lesson is fully available. The Variational
Autoencoder lesson remains a lightweight placeholder for a later phase.
