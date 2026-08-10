"""Trace visual, separator, real-text, and padded-text sequence regions."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        MultimodalSequenceEmbedding,
        make_text_padding_mask,
    )
    from provided import TinyVisionEncoder
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        MultimodalSequenceEmbedding,
        make_text_padding_mask,
    )
    from provided import TinyVisionEncoder


def main() -> None:
    torch.manual_seed(13)
    vision_encoder = TinyVisionEncoder(4, 4, 2, 1, 8, 2, 1)
    model = MultimodalSequenceEmbedding(
        vision_encoder=vision_encoder,
        vocab_size=16,
        max_text_tokens=4,
        embed_dim=8,
    )
    images = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 1, 4, 4)
    text_token_ids = torch.tensor([[4, 5, 0, 0], [7, 8, 9, 0]])
    text_mask = make_text_padding_mask(text_token_ids, pad_token_id=0)
    output = model(images, text_token_ids, text_mask)

    print(f"image shape:              {tuple(images.shape)}")
    print(f"text token shape:         {tuple(text_token_ids.shape)}")
    print(f"unified embedding shape:  {tuple(output.embeddings.shape)}")
    print(f"visual token count:       {output.visual_token_count}")
    print("\nregions:  V=visual, S=separator, T=text")
    print("          ", ["V", "V", "V", "V", "S", "T", "T", "T", "T"])
    print("type IDs: ", output.token_type_ids.tolist())
    print("validity: ", output.attention_mask.tolist())
    print("Padding remains in the tensor but is False in the validity mask.")


if __name__ == "__main__":
    main()
