import pytest
import torch

from implementation import (
    IGNORE_INDEX,
    MultimodalConversation,
    MultimodalSFTBatch,
    SimpleMultimodalTokenizer,
    build_multimodal_conversation,
    collate_multimodal_sft_batch,
    encode_multimodal_example,
    make_toy_multimodal_conversations,
    multimodal_sft_loss,
)
from provided import TinyNativeVLM


def encode_toys(max_text_tokens: int = 9):
    tokenizer = SimpleMultimodalTokenizer()
    examples = [
        encode_multimodal_example(item, tokenizer, max_text_tokens)
        for item in make_toy_multimodal_conversations()
    ]
    return tokenizer, examples


def make_model(vocab_size: int, max_text_tokens: int = 9) -> TinyNativeVLM:
    return TinyNativeVLM(
        image_size=4,
        patch_size=2,
        in_channels=1,
        vocab_size=vocab_size,
        max_text_tokens=max_text_tokens,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
    )


def test_build_conversation_marks_only_assistant_response_and_eos() -> None:
    tokenizer = SimpleMultimodalTokenizer()
    conversation = MultimodalConversation(
        torch.zeros(1, 4, 4), "Describe Image", "Dark Square"
    )

    tokens, assistant_mask = build_multimodal_conversation(conversation, tokenizer)

    assert tokens == [
        "<bos>",
        "<user>",
        "describe",
        "image",
        "<assistant>",
        "dark",
        "square",
        "<eos>",
    ]
    assert assistant_mask == [False, False, False, False, False, True, True, True]


@pytest.mark.parametrize(
    "conversation",
    [
        MultimodalConversation(torch.zeros(4, 4), "prompt", "answer"),
        MultimodalConversation(torch.zeros(1, 4, 4), " ", "answer"),
        MultimodalConversation(torch.zeros(1, 4, 4), "prompt", " "),
    ],
)
def test_build_conversation_rejects_invalid_examples(
    conversation: MultimodalConversation,
) -> None:
    with pytest.raises(ValueError):
        build_multimodal_conversation(conversation, SimpleMultimodalTokenizer())


def test_encode_shifts_labels_and_ignores_prompt() -> None:
    tokenizer = SimpleMultimodalTokenizer()
    conversation = MultimodalConversation(
        torch.zeros(1, 4, 4), "describe image", "dark square"
    )

    example = encode_multimodal_example(conversation, tokenizer, max_text_tokens=7)

    assert example.input_ids.shape == example.labels.shape == (7,)
    assert example.attention_mask.all()
    assert example.labels.tolist()[:4] == [IGNORE_INDEX] * 4
    expected = tokenizer.encode(["dark", "square", "<eos>"])
    assert example.labels.tolist()[4:] == expected


def test_encode_truncates_to_requested_input_width() -> None:
    tokenizer = SimpleMultimodalTokenizer()
    conversation = MultimodalConversation(
        torch.zeros(1, 4, 4), "one two three", "answer continues"
    )

    example = encode_multimodal_example(conversation, tokenizer, max_text_tokens=6)

    assert example.input_ids.numel() == 6
    assert (example.labels != IGNORE_INDEX).sum() == 1


def test_encode_rejects_truncation_before_assistant_targets() -> None:
    tokenizer = SimpleMultimodalTokenizer()
    conversation = MultimodalConversation(
        torch.zeros(1, 4, 4), "one two three four", "answer"
    )

    with pytest.raises(ValueError, match="assistant target"):
        encode_multimodal_example(conversation, tokenizer, max_text_tokens=4)


def test_collate_stacks_images_and_right_pads_text() -> None:
    tokenizer = SimpleMultimodalTokenizer()
    conversations = [
        MultimodalConversation(torch.zeros(1, 4, 4), "color", "black"),
        MultimodalConversation(torch.ones(1, 4, 4), "describe color", "bright white"),
    ]
    examples = [
        encode_multimodal_example(item, tokenizer, max_text_tokens=8)
        for item in conversations
    ]

    batch = collate_multimodal_sft_batch(examples, tokenizer.pad_token_id, 8)

    assert batch.images.shape == (2, 1, 4, 4)
    assert batch.input_ids.shape == batch.labels.shape == (2, 8)
    assert batch.attention_mask.shape == (2, 8)
    assert batch.attention_mask.dtype == torch.bool
    assert not batch.attention_mask[0, -1]
    assert batch.input_ids[0, -1] == tokenizer.pad_token_id
    assert batch.labels[0, -1] == IGNORE_INDEX


def test_collate_rejects_different_image_shapes() -> None:
    tokenizer = SimpleMultimodalTokenizer()
    examples = [
        encode_multimodal_example(
            MultimodalConversation(torch.zeros(1, 4, 4), "p", "a"), tokenizer, 6
        ),
        encode_multimodal_example(
            MultimodalConversation(torch.zeros(1, 6, 6), "p", "a"), tokenizer, 6
        ),
    ]

    with pytest.raises(ValueError, match="same shape"):
        collate_multimodal_sft_batch(examples, tokenizer.pad_token_id, 6)


def test_sft_loss_matches_model_assistant_only_loss() -> None:
    tokenizer, examples = encode_toys()
    batch = collate_multimodal_sft_batch(examples, tokenizer.pad_token_id, 9)
    model = make_model(tokenizer.vocab_size)

    actual = multimodal_sft_loss(model, batch)
    expected = model(
        batch.images, batch.input_ids, batch.attention_mask, batch.labels
    ).loss

    assert expected is not None
    assert torch.allclose(actual, expected)
    assert torch.isfinite(actual)


def test_sft_loss_rejects_batch_without_supervision() -> None:
    tokenizer, examples = encode_toys()
    batch = collate_multimodal_sft_batch(examples, tokenizer.pad_token_id, 9)
    empty = MultimodalSFTBatch(
        batch.images,
        batch.input_ids,
        torch.full_like(batch.labels, IGNORE_INDEX),
        batch.attention_mask,
    )

    with pytest.raises(ValueError, match="supervised"):
        multimodal_sft_loss(make_model(tokenizer.vocab_size), empty)


def test_one_sft_step_reaches_vision_and_text_parameters() -> None:
    tokenizer, examples = encode_toys()
    batch = collate_multimodal_sft_batch(examples, tokenizer.pad_token_id, 9)
    model = make_model(tokenizer.vocab_size)

    multimodal_sft_loss(model, batch).backward()

    assert model.vision_embedding.projection.weight.grad is not None
    assert model.text_embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.vision_embedding.projection.weight.grad.abs().sum() > 0
