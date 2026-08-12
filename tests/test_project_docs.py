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


def test_readme_contains_generic_workspace_workflow() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    commands = [
        "uv sync --locked",
        "uv run rgym list",
        "uv run rgym inspect <lesson_id>",
        "LESSON_ID=llm.05_tiny_gpt",
        'uv run rgym start "$LESSON_ID"',
        'cd "workspace/$LESSON_ID"',
        "uv run rgym test",
        "uv run rgym run",
        "uv run rgym hint",
        "uv run rgym hint --level 2",
        "uv run rgym report",
        'uv run rgym start "$LESSON_ID" --force',
    ]

    assert all(command in readme for command in commands)
    assert "## Start the Bigram Language Model lesson" not in readme
    assert "## Start the Variational Autoencoder lesson" not in readme


def test_repository_contains_no_notebooks() -> None:
    notebooks = [
        path
        for path in PROJECT_ROOT.rglob("*.ipynb")
        if ".venv" not in path.parts and "workspace" not in path.parts
    ]

    assert notebooks == []


def test_mvp_lessons_have_complete_structure() -> None:
    lesson_paths = [
        PROJECT_ROOT / "tracks" / "diffusion" / "01_forward_process",
        PROJECT_ROOT / "tracks" / "diffusion" / "02_noise_prediction",
        PROJECT_ROOT / "tracks" / "diffusion" / "03_ddpm_sampling",
        PROJECT_ROOT / "tracks" / "diffusion" / "04_tiny_unet",
        PROJECT_ROOT / "tracks" / "diffusion" / "05_ddim_sampling",
        PROJECT_ROOT / "tracks" / "diffusion" / "06_classifier_free_guidance",
        PROJECT_ROOT / "tracks" / "diffusion" / "07_latent_diffusion",
        PROJECT_ROOT / "tracks" / "diffusion" / "08_cross_attention_conditioning",
        PROJECT_ROOT / "tracks" / "diffusion" / "09_flow_matching",
        PROJECT_ROOT / "tracks" / "llm" / "01_bigram_lm",
        PROJECT_ROOT / "tracks" / "llm" / "02_tokenizer",
        PROJECT_ROOT / "tracks" / "llm" / "03_causal_attention",
        PROJECT_ROOT / "tracks" / "llm" / "04_transformer_block",
        PROJECT_ROOT / "tracks" / "llm" / "05_tiny_gpt",
        PROJECT_ROOT / "tracks" / "llm" / "06_kv_cache",
        PROJECT_ROOT / "tracks" / "llm" / "07_dpo",
        PROJECT_ROOT / "tracks" / "llm" / "08_sft_data",
        PROJECT_ROOT / "tracks" / "llm" / "09_lora",
        PROJECT_ROOT / "tracks" / "llm" / "10_sampling",
        PROJECT_ROOT / "tracks" / "llm" / "11_vision_patch_embeddings",
        PROJECT_ROOT / "tracks" / "llm" / "12_vision_attention",
        PROJECT_ROOT / "tracks" / "llm" / "13_multimodal_sequence",
        PROJECT_ROOT / "tracks" / "llm" / "14_multimodal_attention_mask",
        PROJECT_ROOT / "tracks" / "llm" / "15_tiny_native_vlm",
        PROJECT_ROOT / "tracks" / "world_models" / "01_vae",
        PROJECT_ROOT / "tracks" / "world_models" / "02_latent_dynamics",
        PROJECT_ROOT / "tracks" / "world_models" / "03_mdn_rnn",
        PROJECT_ROOT / "tracks" / "world_models" / "04_world_model_loop",
        PROJECT_ROOT / "tracks" / "world_models" / "05_cem_planning",
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
