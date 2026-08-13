"""Format, batch, and train tiny image-grounded conversations."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        SimpleMultimodalTokenizer,
        collate_multimodal_sft_batch,
        encode_multimodal_example,
        make_toy_multimodal_conversations,
        multimodal_sft_loss,
    )
    from provided import IGNORE_INDEX, TinyNativeVLM
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        SimpleMultimodalTokenizer,
        collate_multimodal_sft_batch,
        encode_multimodal_example,
        make_toy_multimodal_conversations,
        multimodal_sft_loss,
    )
    from provided import IGNORE_INDEX, TinyNativeVLM


def main() -> None:
    torch.manual_seed(16)
    max_text_tokens = 9
    tokenizer = SimpleMultimodalTokenizer()
    examples = [
        encode_multimodal_example(item, tokenizer, max_text_tokens)
        for item in make_toy_multimodal_conversations()
    ]
    batch = collate_multimodal_sft_batch(
        examples, tokenizer.pad_token_id, max_text_tokens
    )
    model = TinyNativeVLM(
        image_size=4,
        patch_size=2,
        in_channels=1,
        vocab_size=tokenizer.vocab_size,
        max_text_tokens=max_text_tokens,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    print(f"image batch shape: {tuple(batch.images.shape)}")
    print(f"text batch shape:  {tuple(batch.input_ids.shape)}")
    print(f"valid text slots:  {batch.attention_mask.sum(dim=1).tolist()}")
    print(f"supervised slots:  {(batch.labels != IGNORE_INDEX).sum(dim=1).tolist()}")
    print(f"first tokens:      {examples[0].tokens}")
    print(f"first labels:      {batch.labels[0].tolist()}")

    initial_loss = multimodal_sft_loss(model, batch)
    optimizer.zero_grad(set_to_none=True)
    initial_loss.backward()
    vision_gradient = model.vision_embedding.projection.weight.grad
    optimizer.step()
    final_loss = multimodal_sft_loss(model, batch)

    print(f"initial loss:      {initial_loss.item():.4f}")
    print(f"loss after step:   {final_loss.item():.4f}")
    print(f"vision grad > 0:   {vision_gradient.abs().sum().item() > 0}")
    print("Only assistant response targets contribute directly to SFT loss.")


if __name__ == "__main__":
    main()
