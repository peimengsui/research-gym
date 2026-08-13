import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rgym.registry import get_lesson
from rgym.workspace import create_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "lesson_id",
    [
        "diffusion.01_forward_process",
        "diffusion.02_noise_prediction",
        "diffusion.03_ddpm_sampling",
        "diffusion.04_tiny_unet",
        "diffusion.05_ddim_sampling",
        "diffusion.06_classifier_free_guidance",
        "diffusion.07_latent_diffusion",
        "diffusion.08_cross_attention_conditioning",
        "diffusion.09_flow_matching",
        "llm.01_bigram_lm",
        "llm.02_tokenizer",
        "llm.03_causal_attention",
        "llm.04_transformer_block",
        "llm.05_tiny_gpt",
        "llm.06_kv_cache",
        "llm.07_dpo",
        "llm.08_sft_data",
        "llm.09_lora",
        "llm.10_sampling",
        "llm.11_vision_patch_embeddings",
        "llm.12_vision_attention",
        "llm.13_multimodal_sequence",
        "llm.14_multimodal_attention_mask",
        "llm.15_tiny_native_vlm",
        "llm.16_multimodal_sft_data",
        "wm.01_vae",
        "wm.02_latent_dynamics",
        "wm.03_mdn_rnn",
        "wm.04_world_model_loop",
        "wm.05_cem_planning",
    ],
)
def test_solution_passes_lesson_tests(tmp_path: Path, lesson_id: str) -> None:
    lesson = get_lesson(PROJECT_ROOT, lesson_id)
    workspace = create_workspace(lesson, tmp_path)
    shutil.copy2(lesson.path / "solution.py", workspace / lesson.entrypoint)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--quiet"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
