"""Show how chat examples become masked SFT training tensors."""

import shutil
import sys
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        IGNORE_INDEX,
        SimpleChatTokenizer,
        TinyNextTokenLM,
        encode_sft_example,
        make_toy_chats,
        masked_lm_loss,
        pad_sft_batch,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        IGNORE_INDEX,
        SimpleChatTokenizer,
        TinyNextTokenLM,
        encode_sft_example,
        make_toy_chats,
        masked_lm_loss,
        pad_sft_batch,
    )


def show_encoded_example(tokenizer: SimpleChatTokenizer) -> None:
    example = encode_sft_example(make_toy_chats()[0], tokenizer)
    input_tokens = tokenizer.decode_ids(example.input_ids.tolist())
    label_tokens = [
        "<ignore>" if token_id == IGNORE_INDEX else tokenizer.id_to_token[token_id]
        for token_id in example.labels.tolist()
    ]

    print("One encoded SFT example:")
    for index, (input_token, label_token) in enumerate(zip(input_tokens, label_tokens)):
        print(f"  step {index:02d}: input={input_token:>11s} label={label_token}")


def main() -> None:
    torch.manual_seed(3)
    tokenizer = SimpleChatTokenizer()
    examples = [encode_sft_example(chat, tokenizer) for chat in make_toy_chats()]
    batch = pad_sft_batch(examples, tokenizer.pad_token_id)

    show_encoded_example(tokenizer)
    print()
    print(f"batch input_ids shape:      {tuple(batch.input_ids.shape)}")
    print(f"batch labels shape:         {tuple(batch.labels.shape)}")
    print(f"batch attention_mask shape: {tuple(batch.attention_mask.shape)}")

    model = TinyNextTokenLM(vocab_size=tokenizer.vocab_size, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    with torch.no_grad():
        initial_loss = masked_lm_loss(model(batch.input_ids), batch.labels)
    print(f"initial masked LM loss: {initial_loss.item():.3f}")

    for _ in range(80):
        loss = masked_lm_loss(model(batch.input_ids), batch.labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = masked_lm_loss(model(batch.input_ids), batch.labels)
    print(f"final masked LM loss:   {final_loss.item():.3f}")
    print("Only assistant labels contributed to the loss.")


if __name__ == "__main__":
    main()
