"""Learner scaffold for the Supervised Fine-Tuning Data lesson."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

IGNORE_INDEX = -100
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>"]
ROLE_TOKENS = {"user": "<user>", "assistant": "<assistant>"}


@dataclass(frozen=True)
class SFTExample:
    """One encoded SFT example.

    Shapes:
        input_ids: (time,)
        labels: (time,), with ignored positions set to -100
        assistant_mask: (time,), True where labels belong to assistant output
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    assistant_mask: torch.Tensor
    tokens: list[str]


@dataclass(frozen=True)
class SFTBatch:
    """A padded batch of SFT examples.

    Shapes:
        input_ids: (batch, max_time)
        labels: (batch, max_time)
        attention_mask: (batch, max_time)
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


class SimpleChatTokenizer:
    """A tiny whitespace tokenizer with a growing vocabulary."""

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

    def tokenize_text(self, text: str) -> list[str]:
        # TODO: lowercase, strip, and split on whitespace.
        raise NotImplementedError

    def encode_tokens(self, tokens: Sequence[str]) -> list[int]:
        # TODO: convert token strings to ids, adding new text tokens as needed.
        raise NotImplementedError

    def decode_ids(self, ids: Sequence[int]) -> list[str]:
        # TODO: convert ids back to token strings.
        raise NotImplementedError


def validate_messages(messages: Sequence[dict[str, str]]) -> None:
    """Validate a small user/assistant chat for this lesson."""

    # TODO:
    # - require at least one message
    # - allow only roles "user" and "assistant"
    # - require non-empty content
    # - require the first message to be from the user
    # - require at least one assistant message
    raise NotImplementedError


def format_chat(messages: Sequence[dict[str, str]]) -> str:
    """Return a readable role-tagged chat string."""

    # TODO: return lines like "<user> hello" and "<assistant> hi".
    raise NotImplementedError


def build_sft_sequence(
    messages: Sequence[dict[str, str]],
    tokenizer: SimpleChatTokenizer,
) -> tuple[list[str], list[bool]]:
    """Return full tokens and a same-length assistant training mask."""

    # TODO:
    # - start with <bos>, masked False
    # - add a role token for each message, masked False
    # - add content tokens, masked True only for assistant content
    # - add <eos> after assistant messages, masked True
    raise NotImplementedError


def encode_sft_example(
    messages: Sequence[dict[str, str]],
    tokenizer: SimpleChatTokenizer,
) -> SFTExample:
    """Encode one chat into shifted causal-LM inputs and masked labels."""

    # TODO:
    # 1. Build full tokens and train_mask.
    # 2. Convert full tokens to ids.
    # 3. input_ids = full_ids[:-1]
    # 4. labels = full_ids[1:]
    # 5. assistant_mask = train_mask[1:]
    # 6. set labels outside assistant_mask to IGNORE_INDEX.
    raise NotImplementedError


def pad_sft_batch(
    examples: Sequence[SFTExample],
    pad_token_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> SFTBatch:
    """Right-pad encoded examples into batch tensors."""

    # TODO: pad input_ids, labels, and attention_mask to max example length.
    raise NotImplementedError


def masked_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return causal-LM cross-entropy over non-ignored labels only."""

    # TODO: flatten batch/time and call cross_entropy with ignore_index.
    raise NotImplementedError


class TinyNextTokenLM(nn.Module):
    """Tiny next-token model for the demo.

    Input:
        input_ids: (batch, time)

    Output:
        logits: (batch, time, vocab)
    """

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # TODO: embed each token and project to vocab logits.
        raise NotImplementedError


def make_toy_chats() -> list[list[dict[str, str]]]:
    """Return tiny SFT examples for tests and the demo."""

    return [
        [
            {"role": "user", "content": "say hello"},
            {"role": "assistant", "content": "hello friend"},
        ],
        [
            {"role": "user", "content": "say bye"},
            {"role": "assistant", "content": "bye friend"},
        ],
        [
            {"role": "user", "content": "name color"},
            {"role": "assistant", "content": "blue"},
        ],
    ]
