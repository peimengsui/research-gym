"""Reference solution for the Supervised Fine-Tuning Data lesson."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
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
        return text.strip().lower().split()

    def encode_tokens(self, tokens: Sequence[str]) -> list[int]:
        return [self.add_token(token) for token in tokens]

    def decode_ids(self, ids: Sequence[int]) -> list[str]:
        return [self.id_to_token[token_id] for token_id in ids]


def validate_messages(messages: Sequence[dict[str, str]]) -> None:
    """Validate a small user/assistant chat for this lesson."""

    if not messages:
        raise ValueError("messages must contain at least one turn")
    if messages[0].get("role") != "user":
        raise ValueError("the first message must be from the user")

    has_assistant = False
    previous_role: str | None = None
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role not in ROLE_TOKENS:
            raise ValueError("roles must be 'user' or 'assistant'")
        if not content.strip():
            raise ValueError("message content must be non-empty")
        if role == previous_role:
            raise ValueError("user and assistant messages must alternate")
        has_assistant = has_assistant or role == "assistant"
        previous_role = role

    if not has_assistant:
        raise ValueError("messages must contain at least one assistant response")


def format_chat(messages: Sequence[dict[str, str]]) -> str:
    """Return a readable role-tagged chat string."""

    validate_messages(messages)
    lines = ["<bos>"]
    for message in messages:
        role_token = ROLE_TOKENS[message["role"]]
        lines.append(f"{role_token} {message['content'].strip()}")
        if message["role"] == "assistant":
            lines.append("<eos>")
    return "\n".join(lines)


def build_sft_sequence(
    messages: Sequence[dict[str, str]],
    tokenizer: SimpleChatTokenizer,
) -> tuple[list[str], list[bool]]:
    """Return full tokens and a same-length assistant training mask."""

    validate_messages(messages)
    tokens = ["<bos>"]
    train_mask = [False]

    for message in messages:
        role = message["role"]
        tokens.append(ROLE_TOKENS[role])
        train_mask.append(False)

        content_tokens = tokenizer.tokenize_text(message["content"])
        is_assistant = role == "assistant"
        tokens.extend(content_tokens)
        train_mask.extend([is_assistant] * len(content_tokens))

        if is_assistant:
            tokens.append("<eos>")
            train_mask.append(True)

    return tokens, train_mask


def encode_sft_example(
    messages: Sequence[dict[str, str]],
    tokenizer: SimpleChatTokenizer,
) -> SFTExample:
    """Encode one chat into shifted causal-LM inputs and masked labels."""

    tokens, train_mask = build_sft_sequence(messages, tokenizer)
    full_ids = torch.tensor(tokenizer.encode_tokens(tokens), dtype=torch.long)
    if full_ids.numel() < 2:
        raise ValueError("encoded sequence must contain at least two tokens")

    input_ids = full_ids[:-1]
    labels = full_ids[1:].clone()
    assistant_mask = torch.tensor(train_mask[1:], dtype=torch.bool)
    labels[~assistant_mask] = IGNORE_INDEX

    return SFTExample(
        input_ids=input_ids,
        labels=labels,
        assistant_mask=assistant_mask,
        tokens=tokens,
    )


def pad_sft_batch(
    examples: Sequence[SFTExample],
    pad_token_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> SFTBatch:
    """Right-pad encoded examples into batch tensors."""

    if not examples:
        raise ValueError("examples must be non-empty")

    batch_size = len(examples)
    max_time = max(example.input_ids.numel() for example in examples)
    input_ids = torch.full((batch_size, max_time), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_time), ignore_index, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_time), dtype=torch.bool)

    for row, example in enumerate(examples):
        length = example.input_ids.numel()
        input_ids[row, :length] = example.input_ids
        labels[row, :length] = example.labels
        attention_mask[row, :length] = True

    return SFTBatch(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def masked_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return causal-LM cross-entropy over non-ignored labels only."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, time, vocab)")
    if labels.ndim != 2:
        raise ValueError("labels must have shape (batch, time)")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch/time shape")

    batch, time, vocab = logits.shape
    return F.cross_entropy(
        logits.reshape(batch * time, vocab),
        labels.reshape(batch * time),
        ignore_index=ignore_index,
    )


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
        hidden = self.embedding(input_ids)
        return self.lm_head(hidden)


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
