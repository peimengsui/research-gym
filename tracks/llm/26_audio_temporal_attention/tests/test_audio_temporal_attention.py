import pytest
import torch

from implementation import (
    AudioSelfAttention,
    TinyVariableAudioEncoder,
    make_audio_attention_mask,
    make_audio_token_validity,
    safe_masked_softmax,
    valid_stft_frame_counts,
)


def test_valid_stft_frame_counts_handle_short_and_exact_clips() -> None:
    lengths = torch.tensor([0, 5, 6, 10, 18])

    counts = valid_stft_frame_counts(lengths, n_fft=6, hop_length=4)

    assert counts.tolist() == [0, 0, 1, 2, 4]


def test_audio_token_validity_is_time_major_and_requires_full_patches() -> None:
    lengths = torch.tensor([18, 14, 5])

    validity = make_audio_token_validity(lengths, 6, 4, 2, 2, 2)

    assert validity.tolist() == [
        [True, True, True, True],
        [True, True, False, False],
        [False, False, False, False],
    ]


def test_audio_attention_mask_requires_valid_queries_and_keys() -> None:
    validity = torch.tensor([[True, True, False]])

    mask = make_audio_attention_mask(validity)

    assert mask.tolist() == [
        [[True, True, False], [True, True, False], [False, False, False]]
    ]


def test_safe_softmax_returns_zero_for_fully_masked_rows() -> None:
    scores = torch.randn(1, 2, 3, 3)
    mask = make_audio_attention_mask(torch.tensor([[True, True, False]]))

    weights = safe_masked_softmax(scores, mask)

    assert torch.isfinite(weights).all()
    assert torch.equal(weights[:, :, 2], torch.zeros(1, 2, 3))
    assert torch.equal(weights[:, :, :, 2], torch.zeros(1, 2, 3))
    assert torch.allclose(weights[:, :, :2].sum(-1), torch.ones(1, 2, 2))


def test_audio_attention_zeros_invalid_query_outputs() -> None:
    attention = AudioSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)
    validity = torch.tensor([[True, True, False, False], [True, True, True, False]])

    output, weights = attention(x, validity, return_weights=True)

    assert output.shape == x.shape
    assert weights.shape == (2, 2, 4, 4)
    assert torch.equal(output[0, 2:], torch.zeros(2, 8))
    assert torch.equal(output[1, 3:], torch.zeros(1, 8))


def test_encoder_returns_tokens_validity_and_attention_maps() -> None:
    encoder = TinyVariableAudioEncoder(18, 6, 4, 2, 2, 8, 2, 2)
    waveforms = torch.randn(3, 18)
    lengths = torch.tensor([18, 14, 5])

    tokens, validity, maps = encoder(waveforms, lengths, return_weights=True)

    assert tokens.shape == (3, 4, 8)
    assert validity.shape == (3, 4)
    assert len(maps) == 2
    assert maps[0].shape == (3, 2, 4, 4)
    assert torch.equal(tokens[~validity], torch.zeros_like(tokens[~validity]))


def test_valid_outputs_do_not_depend_on_padded_sample_values() -> None:
    torch.manual_seed(26)
    encoder = TinyVariableAudioEncoder(18, 6, 4, 2, 1, 8, 2, 1)
    prefix = torch.randn(14)
    first = torch.cat((prefix, torch.zeros(4)))
    second = torch.cat((prefix, torch.randn(4) * 100))
    waveforms = torch.stack((first, second))
    lengths = torch.tensor([14, 14])

    tokens, validity = encoder(waveforms, lengths)

    assert torch.equal(validity[0], validity[1])
    assert torch.allclose(tokens[0, validity[0]], tokens[1, validity[1]], atol=1e-6)


def test_encoder_supports_gradient_flow_through_valid_audio() -> None:
    encoder = TinyVariableAudioEncoder(18, 6, 4, 2, 2, 8, 2, 1)
    waveforms = torch.randn(2, 18, requires_grad=True)
    lengths = torch.tensor([18, 14])

    tokens, validity = encoder(waveforms, lengths)
    tokens[validity].square().mean().backward()

    assert waveforms.grad is not None
    assert torch.isfinite(waveforms.grad).all()
    assert encoder.blocks[0].attention.query.weight.grad is not None


def test_encoder_rejects_lengths_beyond_padded_width() -> None:
    encoder = TinyVariableAudioEncoder(18, 6, 4, 2, 2, 8, 2, 1)

    with pytest.raises(ValueError, match="cannot exceed"):
        encoder(torch.zeros(1, 18), torch.tensor([19]))
