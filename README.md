# ResearchGym

ResearchGym is a CLI-first learning environment for re-implementing
foundational language-model and world-model ideas from scratch.

It is a guided implementation gym, not a model library. Lessons use ordinary
Python files, focused tests, tiny CPU-friendly demos, and readable reference
implementations. There are no notebooks hiding state or execution order.

## Learning loop

Every lesson follows the same loop:

```text
read the concept
→ inspect the scaffold
→ implement the TODOs
→ run focused tests
→ run a tiny demo
→ request hints when needed
→ reflect on the implementation
→ compare with the reference
→ generate a report
```

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- CPU is sufficient

Install the locked project environment:

```bash
git clone https://github.com/peimengsui/research-gym.git
cd research-gym
uv sync --locked
```

## Discover lessons

```bash
uv run rgym list
uv run rgym inspect llm.01_bigram_lm
uv run rgym inspect llm.02_tokenizer
uv run rgym inspect wm.01_vae
```

Current lesson status:

| Lesson | Status |
| --- | --- |
| `llm.01_bigram_lm` | Complete |
| `llm.02_tokenizer` | Complete |
| `wm.01_vae` | Complete |

## Start the Bigram Language Model lesson

Create a disposable learner workspace:

```bash
uv run rgym start llm.01_bigram_lm
cd workspace/llm.01_bigram_lm
```

Read `concept.md` and `guide.md`, then edit `implementation.py`. The first test
run is expected to fail because the scaffold contains TODOs:

```bash
uv run rgym test
```

Use the complete workspace workflow while learning:

```bash
uv run rgym test
uv run rgym run
uv run rgym hint
uv run rgym hint --level 2
uv run rgym report
```

When your implementation is complete, the tests should pass and the demo
should print a lower final loss than initial loss. Compare your work afterward
with `tracks/llm/01_bigram_lm/solution.py`.

To erase a workspace and restart:

```bash
cd ../..
uv run rgym start llm.01_bigram_lm --force
```

## Start the Tokenizer Fundamentals lesson

From the repository root:

```bash
uv run rgym start llm.02_tokenizer
cd workspace/llm.02_tokenizer
uv run rgym test
```

Implement pair counting, non-overlapping merges, deterministic training,
encoding, decoding, and unknown-character handling in `implementation.py`.

```bash
uv run rgym test
uv run rgym run
uv run rgym hint
uv run rgym report
```

Compare your completed work with
`tracks/llm/02_tokenizer/solution.py` from the repository root.

## Start the Variational Autoencoder lesson

From the repository root:

```bash
uv run rgym start wm.01_vae
cd workspace/wm.01_vae
uv run rgym test
```

Read `concept.md` and `guide.md`, then implement the encoder, latent
distribution, reparameterization trick, decoder, and VAE loss in
`implementation.py`.

Use the same test, demo, hint, and report workflow:

```bash
uv run rgym test
uv run rgym run
uv run rgym hint
uv run rgym report
```

Compare your completed work with
`tracks/world_models/01_vae/solution.py` from the repository root.

## Development

Run repository checks:

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Use
`uv add` or `uv add --dev` when changing them.

## Project documentation

- [Design](docs/design.md): teaching philosophy, lesson anatomy, and AI-tool use
- [Roadmap](docs/roadmap.md): planned lessons and project milestones
- [Execution plan](PLANS.md): detailed MVP phases
- [Contributor and agent guidance](AGENTS.md): repository-wide implementation
  rules

## Contributing

Keep lessons explicit, small, and CPU-friendly. Scaffolds should teach through
TODOs and shapes rather than concealing implementation details. Tests should
check behavior and gradient flow without requiring long training runs.

See `AGENTS.md` before making implementation changes.
