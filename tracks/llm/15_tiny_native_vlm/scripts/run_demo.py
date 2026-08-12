"""Run one end-to-end native VLM training step and inspect its tensor contract."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import TinyNativeVLM, make_next_token_targets
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import TinyNativeVLM, make_next_token_targets


def main() -> None:
    torch.manual_seed(15)
    model = TinyNativeVLM(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        vocab_size=12,
        max_text_tokens=4,
        embed_dim=8,
        num_heads=2,
        num_vision_layers=1,
        num_multimodal_layers=1,
    )
    images = torch.stack((torch.zeros(1, 4, 4), torch.ones(1, 4, 4)))
    # 1=BOS, 4=dark, 5=bright, 2=EOS, 0=PAD
    text_ids = torch.tensor([[1, 4, 2, 0], [1, 5, 2, 0]])
    text_validity = text_ids != 0
    targets = make_next_token_targets(text_ids, text_validity)

    output = model(images, text_ids, text_validity, targets)
    assert output.loss is not None
    output.loss.backward()

    visual_count = model.sequence_embedding.num_visual_tokens
    print(f"image shape:             {tuple(images.shape)}")
    print(f"text input shape:        {tuple(text_ids.shape)}")
    print(f"visual patch tokens:     {visual_count}")
    print(f"unified sequence length: {visual_count + 1 + text_ids.shape[1]}")
    print(f"text-only logits shape:  {tuple(output.logits.shape)}")
    print(f"shifted targets:         {targets.tolist()}")
    print(f"next-token loss:         {output.loss.item():.4f}")
    print(f"attention map shape:     {tuple(output.attention_maps[0].shape)}")
    vision_gradient = (
        model.sequence_embedding.vision_encoder.patch_embedding.projection.weight.grad
    )
    print(f"vision gradient nonzero: {vision_gradient.abs().sum().item() > 0}")
    print(
        "Text supervision now reaches a native image encoder through shared attention."
    )


if __name__ == "__main__":
    main()
