# ResearchGym design

## Purpose

ResearchGym helps learners turn research ideas into working code. It sits
between reading a paper and using a production library: small enough to
understand completely, but structured enough to provide feedback.

The project optimizes for learning clarity rather than training scale,
benchmark performance, or framework coverage.

## Why no notebooks

Notebooks are useful for exploration, but they can hide execution order,
mutable state, and missing dependencies. ResearchGym uses Python and Markdown
files so that:

- the implementation has a clear entry point
- tests run from a clean process
- demos are reproducible commands
- diffs are easy to review
- learners practice the same modules and debugging tools used in real projects

Plots or exploratory notebooks may be useful outside the repository, but a
lesson should not depend on them.

## Why CLI-first

A small CLI gives every lesson one consistent interface:

```bash
rgym list
rgym inspect <lesson_id>
rgym start <lesson_id>
rgym test
rgym run
rgym hint
rgym report
```

This keeps the learning surface stable while lesson content changes. It also
avoids introducing a web application before the core lesson format is proven.

## Lesson structure

Each source lesson lives below `tracks/` and contains:

| File | Purpose |
| --- | --- |
| `lesson.yaml` | Identity, commands, and discovery metadata |
| `README.md` | Short lesson entry point |
| `concept.md` | Research idea and intuition |
| `guide.md` | Ordered implementation steps and common bugs |
| `scaffold.py` | Incomplete learner-facing implementation |
| `solution.py` | Minimal readable reference implementation |
| `hints.md` | Progressive static hints |
| `review.md` | Reflection questions |
| `tests/` | Fast behavioral and shape checks |
| `scripts/run_demo.py` | Tiny reproducible demonstration |

`rgym start` copies the source lesson into `workspace/<lesson_id>/`, renames
`scaffold.py` to `implementation.py`, and omits `solution.py`. Learners can
therefore edit freely without modifying the source lesson.

## Testing philosophy

Tests should be quick enough to run after every small implementation step.
They should emphasize:

- tensor shapes
- basic mathematical behavior
- optional outputs and edge cases
- gradient flow
- preservation of the learner-facing API

Full training quality is demonstrated with a tiny script rather than encoded
as a slow or fragile test.

## Using AI coding tools

AI tools should support the learning loop, not bypass it.

Good uses include:

- explaining a tensor shape or error message
- reviewing a learner's implementation
- suggesting a focused debugging experiment
- comparing two implementation choices
- asking questions that reveal a misconception

Poor uses include:

- replacing the scaffold with the complete solution before attempting it
- generating opaque abstractions that hide the research idea
- changing tests only to make incorrect code pass
- adding large dependencies to avoid understanding a small operation

When an AI tool contributes code, the learner should still be able to explain
each tensor operation, run the tests, and describe why the implementation
matches the concept.

## Scope and evolution

The first version is deliberately CPU-friendly, local, and single-process.
Future work should deepen the lesson catalog before adding infrastructure.
Potential later capabilities include richer review reports, optional visual
outputs, lesson authoring checks, and carefully scoped interactive interfaces.

See [roadmap.md](roadmap.md) for the planned sequence.
