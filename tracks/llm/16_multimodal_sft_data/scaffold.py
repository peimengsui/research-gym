"""Learner scaffold for multimodal supervised fine-tuning data.

TinyNativeVLM is complete in provided.py. TODOs focus on turning an image,
user prompt, and assistant response into assistant-only training supervision.
"""

from dataclasses import dataclass
from typing import Sequence

import torch

from provided import IGNORE_INDEX, TinyNativeVLM  # noqa: F401 - used in TODOs


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
    image: torch.Tensor
    user_text: str
    assistant_text: str


@dataclass(frozen=True)
class EncodedMultimodalExample:
    image: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    tokens: list[str]


@dataclass(frozen=True)
class MultimodalSFTBatch:
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
    # TODO 1: Format:
    # [<bos>, <user>, user words..., <assistant>, assistant words..., <eos>]
    # Return a same-length bool list that is True only for assistant words and
    # <eos>. Role markers, BOS, and user words are conditioning, not targets.
    raise NotImplementedError


def encode_multimodal_example(
    conversation: MultimodalConversation,
    tokenizer: SimpleMultimodalTokenizer,
    max_text_tokens: int,
) -> EncodedMultimodalExample:
    """Shift and truncate one conversation into VLM inputs and labels."""

    if max_text_tokens <= 0:
        raise ValueError("max_text_tokens must be positive")
    # TODO 2: Build tokens/mask, truncate both to max_text_tokens + 1, encode,
    # and shift into input_ids=full_ids[:-1], labels=full_ids[1:]. Apply the
    # correspondingly shifted assistant mask to labels with IGNORE_INDEX.
    # Reject truncation that leaves no assistant target. attention_mask is all
    # True here because per-example padding has not happened yet.
    raise NotImplementedError


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
    # TODO 3: Stack images. Create fixed (batch, max_text_tokens) tensors:
    # input_ids filled with pad_token_id, labels with IGNORE_INDEX, and a False
    # boolean attention_mask. Copy each variable-length example into the left.
    raise NotImplementedError


def multimodal_sft_loss(
    model: TinyNativeVLM,
    batch: MultimodalSFTBatch,
) -> torch.Tensor:
    """Run the VLM and return assistant-only next-token cross-entropy."""

    # TODO 4: Reject a batch with no non-ignored labels. Pass images, input IDs,
    # attention mask, and labels to model. Verify loss is present and return it.
    raise NotImplementedError


def make_toy_multimodal_conversations() -> list[MultimodalConversation]:
    """Return tiny synthetic image-caption examples for the demo."""

    # TODO 5: Return two 1 x 4 x 4 image conversations: an all-zero dark image
    # and an all-one bright image, sharing the user prompt "what brightness" and
    # using assistant responses "dark image" and "bright image" respectively.
    raise NotImplementedError
