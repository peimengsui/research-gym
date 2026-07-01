import pytest
import torch
import torch.nn.functional as F

from implementation import (
    IGNORE_INDEX,
    SimpleChatTokenizer,
    TinyNextTokenLM,
    build_sft_sequence,
    encode_sft_example,
    format_chat,
    make_toy_chats,
    masked_lm_loss,
    pad_sft_batch,
    validate_messages,
)


def test_format_chat_adds_role_tags_and_eos() -> None:
    messages = [
        {"role": "user", "content": "Say Hello"},
        {"role": "assistant", "content": "Hello friend"},
    ]

    formatted = format_chat(messages)

    assert formatted.splitlines() == [
        "<bos>",
        "<user> Say Hello",
        "<assistant> Hello friend",
        "<eos>",
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "hello"}],
        [{"role": "system", "content": "be nice"}],
        [{"role": "user", "content": "   "}],
        [{"role": "user", "content": "hello"}],
        [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "hi"},
        ],
    ],
)
def test_validate_messages_rejects_invalid_chats(
    messages: list[dict[str, str]],
) -> None:
    with pytest.raises(ValueError):
        validate_messages(messages)


def test_build_sft_sequence_marks_only_assistant_tokens() -> None:
    tokenizer = SimpleChatTokenizer()
    messages = [
        {"role": "user", "content": "Say Hello"},
        {"role": "assistant", "content": "Hello friend"},
    ]

    tokens, train_mask = build_sft_sequence(messages, tokenizer)

    assert tokens == [
        "<bos>",
        "<user>",
        "say",
        "hello",
        "<assistant>",
        "hello",
        "friend",
        "<eos>",
    ]
    assert train_mask == [False, False, False, False, False, True, True, True]


def test_encode_sft_example_shifts_and_masks_labels() -> None:
    tokenizer = SimpleChatTokenizer()
    messages = [
        {"role": "user", "content": "say hello"},
        {"role": "assistant", "content": "hello friend"},
    ]

    encoded = encode_sft_example(messages, tokenizer)
    label_tokens = [
        "<user>",
        "say",
        "hello",
        "<assistant>",
        "hello",
        "friend",
        "<eos>",
    ]
    expected_labels = torch.tensor(tokenizer.encode_tokens(label_tokens))
    expected_labels[:4] = IGNORE_INDEX

    assert encoded.input_ids.shape == encoded.labels.shape == (7,)
    assert encoded.assistant_mask.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert torch.equal(encoded.labels, expected_labels)


def test_pad_sft_batch_right_pads_inputs_labels_and_attention() -> None:
    tokenizer = SimpleChatTokenizer()
    examples = [
        encode_sft_example(messages, tokenizer)
        for messages in [
            [
                {"role": "user", "content": "say hello"},
                {"role": "assistant", "content": "hello friend"},
            ],
            [
                {"role": "user", "content": "color"},
                {"role": "assistant", "content": "blue"},
            ],
        ]
    ]

    batch = pad_sft_batch(examples, tokenizer.pad_token_id)

    assert batch.input_ids.shape == (2, 7)
    assert batch.labels.shape == (2, 7)
    assert batch.attention_mask.dtype == torch.bool
    assert batch.attention_mask[0].all()
    assert batch.attention_mask[1].tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert batch.input_ids[1, -1] == tokenizer.pad_token_id
    assert batch.labels[1, -1] == IGNORE_INDEX


def test_masked_lm_loss_ignores_prompt_and_padding_labels() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ]
    )
    labels = torch.tensor([[IGNORE_INDEX, 1, 2]])

    actual = masked_lm_loss(logits, labels)
    expected = F.cross_entropy(
        logits[:, 1:, :].reshape(2, 3),
        torch.tensor([1, 2]),
    )

    assert torch.allclose(actual, expected)


def test_tiny_model_can_fit_toy_assistant_labels() -> None:
    torch.manual_seed(0)
    tokenizer = SimpleChatTokenizer()
    examples = [encode_sft_example(chat, tokenizer) for chat in make_toy_chats()]
    batch = pad_sft_batch(examples, tokenizer.pad_token_id)
    model = TinyNextTokenLM(vocab_size=tokenizer.vocab_size, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    with torch.no_grad():
        initial_loss = masked_lm_loss(model(batch.input_ids), batch.labels)

    for _ in range(80):
        loss = masked_lm_loss(model(batch.input_ids), batch.labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = masked_lm_loss(model(batch.input_ids), batch.labels)

    assert final_loss < initial_loss * 0.25
    assert torch.isfinite(final_loss)
