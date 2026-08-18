"""Compare cached multimodal generation with full-sequence recomputation."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import TinyCachedVLM, generate_multimodal
    from provided import TinyVocabulary, make_toy_generation_inputs
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import TinyCachedVLM, generate_multimodal
    from provided import TinyVocabulary, make_toy_generation_inputs


def generate_with_full_recomputation(
    model: TinyCachedVLM,
    images: torch.Tensor,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    generated = prompt
    for _ in range(max_new_tokens):
        logits = model.forward_full(images, generated)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return generated


def main() -> None:
    torch.manual_seed(17)
    vocabulary = TinyVocabulary()
    images, prompt = make_toy_generation_inputs(vocabulary)
    model = TinyCachedVLM(
        image_size=4,
        patch_size=2,
        in_channels=1,
        vocab_size=len(vocabulary.tokens),
        max_text_tokens=9,
        embed_dim=8,
        num_heads=2,
        num_layers=2,
    )

    _, prefill_caches = model.prefill(images, prompt)
    calls = 0

    def count_vision_calls(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        del module, inputs, output
        nonlocal calls
        calls += 1

    handle = model.vision_embedding.register_forward_hook(count_vision_calls)
    cached = generate_multimodal(model, images, prompt, max_new_tokens=4)
    handle.remove()
    recomputed = generate_with_full_recomputation(model, images, prompt, 4)

    print(f"visual tokens:       {model.num_visual_tokens}")
    print(f"prompt tokens:       {prompt.shape[1]}")
    print(f"prefill cache shape: {tuple(prefill_caches[0][0].shape)}")
    print(f"vision prefill calls:{calls:>3}")
    print(f"cached == full:      {torch.equal(cached, recomputed)}")
    for row, decoded in enumerate(vocabulary.decode(cached)):
        print(f"row {row}: {' '.join(decoded)}")
    print("The image and prompt were encoded once; decode steps reused their KV cache.")


if __name__ == "__main__":
    main()
