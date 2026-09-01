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

## Stage 1 MVP goal

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
    diffusion/
      01_forward_process/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_forward_process.py
        scripts/
          run_demo.py

      02_noise_prediction/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_noise_prediction.py
        scripts/
          run_demo.py

      03_ddpm_sampling/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_ddpm_sampling.py
        scripts/
          run_demo.py

      04_tiny_unet/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_tiny_unet.py
        scripts/
          run_demo.py

      05_ddim_sampling/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_ddim_sampling.py
        scripts/
          run_demo.py

      06_classifier_free_guidance/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_classifier_free_guidance.py
        scripts/
          run_demo.py

      07_latent_diffusion/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_latent_diffusion.py
        scripts/
          run_demo.py

      08_cross_attention_conditioning/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_cross_attention_conditioning.py
        scripts/
          run_demo.py

      09_flow_matching/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_flow_matching.py
        scripts/
          run_demo.py

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

      03_causal_attention/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_causal_attention.py
        scripts/
          run_demo.py

      04_transformer_block/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_transformer_block.py
        scripts/
          run_demo.py

      05_tiny_gpt/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_tiny_gpt.py
        scripts/
          run_demo.py

      06_kv_cache/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_kv_cache.py
        scripts/
          run_demo.py

      07_dpo/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_dpo.py
        scripts/
          run_demo.py

      08_sft_data/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_sft_data.py
        scripts/
          run_demo.py

      09_lora/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_lora.py
        scripts/
          run_demo.py

      10_sampling/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_sampling.py
        scripts/
          run_demo.py

      11_vision_patch_embeddings/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_vision_patch_embeddings.py
        scripts/
          run_demo.py

      12_vision_attention/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_vision_attention.py
        scripts/
          run_demo.py

      13_multimodal_sequence/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_multimodal_sequence.py
        scripts/
          run_demo.py

      14_multimodal_attention_mask/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_multimodal_attention_mask.py
        scripts/
          run_demo.py

      15_tiny_native_vlm/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_tiny_native_vlm.py
        scripts/
          run_demo.py

      16_multimodal_sft_data/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_multimodal_sft_data.py
        scripts/
          run_demo.py

      17_multimodal_generation/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_multimodal_generation.py
        scripts/
          run_demo.py

      18_multimodal_eval/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_multimodal_eval.py
        scripts/
          run_demo.py

      19_quantization/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_quantization.py
        scripts/
          run_demo.py

      20_speculative_decoding/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_speculative_decoding.py
        scripts/
          run_demo.py

      21_paged_kv_and_continuous_batching/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_paged_kv_and_continuous_batching.py
        scripts/
          run_demo.py

      22_video_tubelet_embeddings/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_video_tubelet_embeddings.py
        scripts/
          run_demo.py

      23_factorized_video_attention/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_factorized_video_attention.py
        scripts/
          run_demo.py

      24_tiny_video_language_model/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_tiny_video_language_model.py
        scripts/
          run_demo.py

      25_audio_spectrogram_tokens/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_audio_spectrogram_tokens.py
        scripts/
          run_demo.py

      26_audio_temporal_attention/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_audio_temporal_attention.py
        scripts/
          run_demo.py

      27_tiny_audio_language_model/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_tiny_audio_language_model.py
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

      02_latent_dynamics/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_latent_dynamics.py
        scripts/
          run_demo.py

      03_mdn_rnn/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_mdn_rnn.py
        scripts/
          run_demo.py

      04_world_model_loop/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_world_model_loop.py
        scripts/
          run_demo.py

      05_cem_planning/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        hints.md
        review.md
        tests/
          test_cem_planning.py
        scripts/
          run_demo.py

      06_jepa_latent_prediction/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_jepa_latent_prediction.py
        scripts/
          run_demo.py

      07_action_conditioned_jepa/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_action_conditioned_jepa.py
        scripts/
          run_demo.py

      08_imagined_rollouts/
        lesson.yaml
        README.md
        concept.md
        guide.md
        scaffold.py
        solution.py
        provided.py
        hints.md
        review.md
        tests/
          test_imagined_rollouts.py
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
