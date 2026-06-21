# PLANS.md

# ResearchGym MVP Execution Plan: Week 1–2

## Objective

Build the first CLI-first MVP of ResearchGym.

The MVP should let a learner:

1. discover available lessons
2. inspect a lesson
3. start a lesson into a workspace
4. edit `implementation.py`
5. run tests
6. run a tiny demo
7. request static hints
8. generate a Markdown report

The MVP should avoid notebooks and web UI.

The first two lessons are:

* `llm.01_bigram_lm`
* `wm.01_vae`

The MVP should be CPU-friendly and testable in a normal development environment.

---

# Phase 0: Repository foundation

## Goal

Create the Python package, CLI entrypoint, test setup, and project README.

## Tasks

### 0.1 Create project files

Create:

```text
pyproject.toml
uv.lock
README.md
rgym/__init__.py
rgym/cli.py
rgym/lesson.py
rgym/registry.py
rgym/workspace.py
rgym/runner.py
rgym/report.py
tests/test_registry.py
tests/test_cli_smoke.py
```

### 0.2 Configure dependencies

Use Python 3.11+ and `uv` for Python version, virtual environment, dependency, and lockfile management.

Initialize and synchronize the project with:

```bash
uv init
uv add torch typer pyyaml rich
uv add --dev pytest ruff
uv sync
```

Runtime dependencies:

```text
torch
typer
pyyaml
rich
```

Development dependencies:

```text
pytest
ruff
```

Declare dependencies in `pyproject.toml` and commit `uv.lock`. Use `uv add` and `uv remove` for dependency changes. Do not use `pip install`, manually managed virtual environments, Poetry, Pipenv, or Conda for the project workflow.

### 0.3 Configure CLI entrypoint

`pyproject.toml` should expose:

```toml
[project.scripts]
rgym = "rgym.cli:app"
```

### 0.4 Create minimal README

README should explain:

* what ResearchGym is
* why it avoids notebooks
* how to install locally
* how to list lessons
* how to start a lesson
* how to run tests and demos

Example commands:

```bash
uv sync
uv run rgym list
uv run rgym inspect llm.01_bigram_lm
uv run rgym start llm.01_bigram_lm
cd workspace/llm.01_bigram_lm
uv run rgym test
uv run rgym run
```

## Acceptance criteria

* `uv sync` works
* `uv run rgym --help` works
* `uv run pytest` runs
* `uv.lock` is committed and up to date
* README contains the basic workflow

---

# Phase 1: Lesson registry and metadata

## Goal

Implement a registry that discovers lessons from `tracks/**/lesson.yaml`.

## Tasks

### 1.1 Define lesson metadata model

Create a simple dataclass in `rgym/lesson.py`:

Fields:

```python
id: str
title: str
track: str
level: str
summary: str
path: Path
entrypoint: str
test_command: str
run_command: str
```

### 1.2 Implement registry discovery

In `rgym/registry.py`, implement:

```python
discover_lessons(root: Path) -> list[Lesson]
get_lesson(root: Path, lesson_id: str) -> Lesson
```

Discovery should find:

```text
tracks/**/lesson.yaml
```

### 1.3 Implement `rgym list`

Expected output should group or at least show:

```text
llm.01_bigram_lm        Bigram Language Model
wm.01_vae               Variational Autoencoder
```

### 1.4 Implement `rgym inspect <lesson_id>`

Should print:

* lesson id
* title
* track
* level
* summary
* lesson path
* commands
* available docs

## Acceptance criteria

* `rgym list` shows both MVP lessons
* `rgym inspect llm.01_bigram_lm` works
* `rgym inspect wm.01_vae` works
* registry tests pass

---

# Phase 2: Workspace creation

## Goal

Allow learners to start a lesson into an editable workspace.

## Tasks

### 2.1 Implement workspace copy

Command:

```bash
rgym start <lesson_id>
```

Should create:

```text
workspace/<lesson_id>/
```

Copy lesson files into the workspace.

Special rule:

```text
scaffold.py -> implementation.py
```

Keep docs, tests, scripts, and metadata.

For MVP, do not worry about hiding `solution.py`. It can either be copied or skipped. Prefer skipping it from workspace and leaving it in source.

### 2.2 Avoid accidental overwrite

If workspace already exists, default behavior should refuse to overwrite.

Show message:

```text
Workspace already exists: workspace/<lesson_id>
Use --force to recreate it.
```

Optional:

```bash
rgym start <lesson_id> --force
```

### 2.3 Make workspace self-contained

After starting, the user should be able to run:

```bash
cd workspace/llm.01_bigram_lm
uv run rgym test
uv run rgym run
```

The workspace tests should import from:

```python
from implementation import ...
```

## Acceptance criteria

* `rgym start llm.01_bigram_lm` creates a workspace
* `implementation.py` exists
* source lesson files are unchanged
* duplicate start does not overwrite unless `--force` is used

---

# Phase 3: Runner commands

## Goal

Implement commands that work from inside a lesson workspace.

## Tasks

### 3.1 Implement `rgym test`

When run inside a workspace, it should read `lesson.yaml` and execute the test command.

Default:

```bash
uv run pytest tests
```

Use `subprocess.run`. Lesson commands should execute through `uv run` so they use the locked project environment.

Return non-zero exit code if tests fail.

### 3.2 Implement `rgym run`

When run inside a workspace, it should read `lesson.yaml` and execute the run command.

Default:

```bash
uv run python scripts/run_demo.py
```

### 3.3 Implement `rgym hint`

Read `hints.md` and print the first hint section.

Optional:

```bash
uv run rgym hint --level 2
```

For MVP, static hints are enough. No LLM API integration.

### 3.4 Implement `rgym report`

Generate:

```text
report.md
```

Include:

* lesson id
* title
* timestamp
* files present
* latest test/run command
* static reflection questions from `review.md` if available

It is okay if the first report is simple.

## Acceptance criteria

From inside each workspace:

```bash
uv run rgym test
uv run rgym run
uv run rgym hint
uv run rgym report
```

all work.

---

# Phase 4: Lesson 1 — Bigram Language Model

## Goal

Create the first LLM lesson.

## Source path

```text
tracks/llm/01_bigram_lm/
```

## Files

Create:

```text
lesson.yaml
README.md
concept.md
guide.md
scaffold.py
solution.py
hints.md
review.md
tests/test_bigram.py
scripts/run_demo.py
```

## Learning objectives

The learner should implement:

* token embedding table
* forward pass returning logits
* cross-entropy loss
* generate function
* simple autoregressive sampling

## Scaffold requirements

`scaffold.py` should define:

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        ...

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        ...
```

Scaffold should include TODO comments and expected shapes.

Expected shapes:

```text
idx: [batch, time]
targets: [batch, time]
logits: [batch, time, vocab_size]
loss: scalar
```

## Tests

Tests should verify:

1. model instantiates
2. forward returns logits shape `[B, T, vocab_size]`
3. loss is scalar when targets are provided
4. loss is `None` when targets are omitted
5. generation extends sequence length by `max_new_tokens`
6. gradients flow to embedding parameters

Tests should be fast and CPU-friendly.

## Demo script

`scripts/run_demo.py` should:

* create a tiny text corpus inline
* build a character vocabulary
* train for a very small number of steps
* print initial/final loss
* print a generated sample

Keep runtime short.

## Acceptance criteria

Inside workspace:

```bash
uv run rgym test
uv run rgym run
```

should work after the learner implements `implementation.py`.

Source `solution.py` should pass tests if copied to `implementation.py`.

---

# Phase 5: Lesson 2 — Variational Autoencoder

## Goal

Create the first world-model lesson.

## Source path

```text
tracks/world_models/01_vae/
```

## Files

Create:

```text
lesson.yaml
README.md
concept.md
guide.md
scaffold.py
solution.py
hints.md
review.md
tests/test_vae.py
scripts/run_demo.py
```

## Learning objectives

The learner should implement:

* encoder
* latent mean and log variance
* reparameterization trick
* decoder
* reconstruction loss
* KL loss
* total VAE loss

## Scaffold requirements

`scaffold.py` should define:

```python
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        ...

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        ...

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

Use flattened vectors for MVP, not convolutional images.

Expected shapes:

```text
x: [batch, input_dim]
mu: [batch, latent_dim]
logvar: [batch, latent_dim]
z: [batch, latent_dim]
recon_x: [batch, input_dim]
```

## Tests

Tests should verify:

1. model instantiates
2. encode returns correct shapes
3. reparameterize returns correct shape
4. decode returns correct shape
5. forward returns reconstruction, mu, logvar
6. loss returns total, reconstruction, KL terms
7. gradients flow through the model

Tests should be fast and CPU-friendly.

## Demo script

`scripts/run_demo.py` should:

* create synthetic binary vectors or simple generated toy data
* train for a tiny number of steps
* print reconstruction loss and KL loss
* save optional output to `outputs/`

Avoid downloading datasets in MVP.

## Acceptance criteria

Inside workspace:

```bash
uv run rgym test
uv run rgym run
```

should work after the learner implements `implementation.py`.

Source `solution.py` should pass tests if copied to `implementation.py`.

---

# Phase 6: Polish and verification

## Goal

Make the MVP pleasant enough to open source.

## Tasks

### 6.1 Add root README workflow

README should include:

```bash
uv sync
uv run rgym list
uv run rgym inspect llm.01_bigram_lm
uv run rgym start llm.01_bigram_lm
cd workspace/llm.01_bigram_lm
uv run rgym test
uv run rgym run
uv run rgym hint
uv run rgym report
```

### 6.2 Add design note

Create:

```text
docs/design.md
```

Explain:

* why no notebooks
* why CLI-first
* how lessons are structured
* how AI coding tools should be used
* future roadmap

### 6.3 Add roadmap

Create:

```text
docs/roadmap.md
```

Include:

* Week 3–4: tokenizer, causal attention, latent dynamics, MDN-RNN
* Month 2: tiny GPT, CEM planning, KV cache
* Month 3+: Dreamer-lite, DPO, VLM mini-lab

### 6.4 Run full verification

Run:

```bash
uv sync --locked
uv run pytest
uv run rgym list
uv run rgym inspect llm.01_bigram_lm
uv run rgym inspect wm.01_vae
rm -rf workspace
uv run rgym start llm.01_bigram_lm
cd workspace/llm.01_bigram_lm
uv run rgym test
uv run rgym run
uv run rgym hint
uv run rgym report
cd ../..
uv run rgym start wm.01_vae
cd workspace/wm.01_vae
uv run rgym test
uv run rgym run
uv run rgym hint
uv run rgym report
```

## Final acceptance criteria

The MVP is complete when:

* `uv sync --locked` installs the package and dependencies locally
* `uv.lock` is committed and matches `pyproject.toml`
* CLI works
* both lessons can be listed and inspected
* both lessons can be started into workspace
* both lessons contain scaffold, guide, tests, demo, hints, review
* tests are CPU-friendly
* no notebooks exist
* README explains the learning loop
* docs explain design and roadmap
