"""Reference solution for multimodal supervised fine-tuning data."""

from dataclasses import dataclass
from typing import Sequence

import torch

from provided import IGNORE_INDEX, TinyNativeVLM


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>"]


class SimpleMultimodalTokenizer:
    """Tiny lowercase whitespace tokenizer with explicit conversation tokens."""

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        for token in SPECIAL_TOKENS:
            self.add_token(token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def tokenize(self, text: str) -> list[str]:
        return text.strip().lower().split()

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.add_token(token) for token in tokens]


@dataclass(frozen=True)
class MultimodalConversation:
    """One image-grounded user prompt and assistant response."""

    image: torch.Tensor
    user_text: str
    assistant_text: str


@dataclass(frozen=True)
class EncodedMultimodalExample:
    """One variable-length encoded example before batch padding."""

    image: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    tokens: list[str]


@dataclass(frozen=True)
class MultimodalSFTBatch:
    """Fixed-width images, text inputs, validity, and assistant-only labels."""

    images: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


def build_multimodal_conversation(
    conversation: MultimodalConversation,
    tokenizer: SimpleMultimodalTokenizer,
) -> tuple[list[str], list[bool]]:
    """Return formatted tokens and a same-length assistant-target mask."""

    if conversation.image.ndim != 3:
        raise ValueError("image must have shape (channels, height, width)")
    if not conversation.user_text.strip() or not conversation.assistant_text.strip():
        raise ValueError("user and assistant text must be non-empty")

    user_tokens = tokenizer.tokenize(conversation.user_text)
    assistant_tokens = tokenizer.tokenize(conversation.assistant_text)
    tokens = (
        ["<bos>", "<user>"]
        + user_tokens
        + ["<assistant>"]
        + assistant_tokens
        + ["<eos>"]
    )
    assistant_target_mask = [False] * (2 + len(user_tokens) + 1) + [True] * (
        len(assistant_tokens) + 1
    )
    return tokens, assistant_target_mask


def encode_multimodal_example(
    conversation: MultimodalConversation,
    tokenizer: SimpleMultimodalTokenizer,
    max_text_tokens: int,
) -> EncodedMultimodalExample:
    """Shift and truncate one conversation into VLM inputs and labels."""

    if max_text_tokens <= 0:
        raise ValueError("max_text_tokens must be positive")
    tokens, assistant_target_mask = build_multimodal_conversation(
        conversation, tokenizer
    )
    # Need at most max_text_tokens inputs plus one shifted target token.
    tokens = tokens[: max_text_tokens + 1]
    assistant_target_mask = assistant_target_mask[: max_text_tokens + 1]
    if len(tokens) < 2:
        raise ValueError("encoded conversation must contain at least two tokens")

    full_ids = torch.tensor(tokenizer.encode(tokens), dtype=torch.long)
    input_ids = full_ids[:-1]
    labels = full_ids[1:].clone()
    target_mask = torch.tensor(assistant_target_mask[1:], dtype=torch.bool)
    labels[~target_mask] = IGNORE_INDEX
    if not target_mask.any():
        raise ValueError("truncation removed every assistant target")

    return EncodedMultimodalExample(
        image=conversation.image,
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        tokens=tokens,
    )


def collate_multimodal_sft_batch(
    examples: Sequence[EncodedMultimodalExample],
    pad_token_id: int,
    max_text_tokens: int,
) -> MultimodalSFTBatch:
    """Stack images and right-pad text fields to one fixed width."""

    if not examples:
        raise ValueError("examples must be non-empty")
    if max_text_tokens <= 0:
        raise ValueError("max_text_tokens must be positive")
    if any(example.input_ids.numel() > max_text_tokens for example in examples):
        raise ValueError("an example exceeds max_text_tokens")
    image_shape = examples[0].image.shape
    if any(example.image.shape != image_shape for example in examples):
        raise ValueError("all images must have the same shape")

    batch_size = len(examples)
    input_ids = torch.full(
        (batch_size, max_text_tokens), pad_token_id, dtype=torch.long
    )
    labels = torch.full((batch_size, max_text_tokens), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_text_tokens), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = example.input_ids.numel()
        input_ids[row, :length] = example.input_ids
        labels[row, :length] = example.labels
        attention_mask[row, :length] = example.attention_mask

    return MultimodalSFTBatch(
        images=torch.stack([example.image for example in examples]),
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


def multimodal_sft_loss(
    model: TinyNativeVLM,
    batch: MultimodalSFTBatch,
) -> torch.Tensor:
    """Run the VLM and return assistant-only next-token cross-entropy."""

    if not (batch.labels != IGNORE_INDEX).any():
        raise ValueError("batch must contain at least one supervised label")
    output = model(
        batch.images,
        batch.input_ids,
        batch.attention_mask,
        batch.labels,
    )
    if output.loss is None:
        raise RuntimeError("model did not return a loss")
    return output.loss


def make_toy_multimodal_conversations() -> list[MultimodalConversation]:
    """Return tiny synthetic image-caption examples for the demo."""

    return [
        MultimodalConversation(
            image=torch.zeros(1, 4, 4),
            user_text="what brightness",
            assistant_text="dark image",
        ),
        MultimodalConversation(
            image=torch.ones(1, 4, 4),
            user_text="what brightness",
            assistant_text="bright image",
        ),
    ]
