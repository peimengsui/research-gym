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
        "llm.01_bigram_lm",
        "llm.02_tokenizer",
        "wm.01_vae",
        "wm.02_latent_dynamics",
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
