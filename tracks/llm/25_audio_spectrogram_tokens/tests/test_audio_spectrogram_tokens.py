import pytest
import torch

from implementation import (
    AudioSpectrogramEmbedding,
    spectrogram_patchify,
    spectrogram_unpatchify,
    waveform_to_log_spectrogram,
)


def test_log_spectrogram_has_expected_shape_and_nonnegative_values() -> None:
    waveforms = torch.randn(2, 18)

    spectrograms = waveform_to_log_spectrogram(waveforms, n_fft=6, hop_length=4)

    assert spectrograms.shape == (2, 4, 4)
    assert (spectrograms >= 0).all()


def test_sine_wave_has_largest_average_magnitude_near_its_bin() -> None:
    sample_index = torch.arange(40, dtype=torch.float32)
    waveform = torch.sin(2 * torch.pi * 2 * sample_index / 16).unsqueeze(0)

    spectrogram = waveform_to_log_spectrogram(waveform, n_fft=16, hop_length=8)

    assert spectrogram.mean(dim=-1).argmax().item() == 2


def test_spectrogram_patch_order_is_time_then_frequency() -> None:
    spectrogram = torch.arange(4 * 4, dtype=torch.float32).reshape(1, 4, 4)

    patches = spectrogram_patchify(spectrogram, 2, 2)

    assert patches.shape == (1, 4, 4)
    assert patches[0, :, 0].tolist() == [0.0, 8.0, 2.0, 10.0]
    assert patches[0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]


def test_patchify_and_unpatchify_round_trip() -> None:
    spectrograms = torch.randn(3, 6, 8)

    patches = spectrogram_patchify(spectrograms, 2, 4)
    reconstructed = spectrogram_unpatchify(patches, 6, 8, 2, 4)

    assert torch.equal(reconstructed, spectrograms)


def test_patchify_rejects_nondivisible_axes() -> None:
    with pytest.raises(ValueError, match="frequency bins"):
        spectrogram_patchify(torch.zeros(1, 5, 4), 2, 2)
    with pytest.raises(ValueError, match="time frames"):
        spectrogram_patchify(torch.zeros(1, 4, 5), 2, 2)


def test_embedding_returns_expected_shape_and_metadata() -> None:
    embedding = AudioSpectrogramEmbedding(18, 6, 4, 2, 2, 8)

    tokens = embedding(torch.randn(3, 18))

    assert tokens.shape == (3, 4, 8)
    assert embedding.frequency_token_count == 2
    assert embedding.temporal_token_count == 2
    assert embedding.num_tokens == 4


def test_factorized_positions_follow_time_major_order() -> None:
    embedding = AudioSpectrogramEmbedding(18, 6, 4, 2, 2, 1)
    with torch.no_grad():
        embedding.projection.weight.zero_()
        embedding.projection.bias.zero_()
        embedding.temporal_position_embedding.weight.copy_(
            torch.tensor([[0.0], [10.0]])
        )
        embedding.frequency_position_embedding.weight.copy_(
            torch.tensor([[0.0], [1.0]])
        )

    tokens = embedding(torch.zeros(1, 18))

    assert tokens[0, :, 0].tolist() == [0.0, 1.0, 10.0, 11.0]


def test_embedding_rejects_wrong_sample_count() -> None:
    embedding = AudioSpectrogramEmbedding(18, 6, 4, 2, 2, 8)

    with pytest.raises(ValueError, match="shape"):
        embedding(torch.zeros(2, 17))


def test_embedding_supports_waveform_and_parameter_gradients() -> None:
    embedding = AudioSpectrogramEmbedding(18, 6, 4, 2, 2, 8)
    waveforms = torch.randn(2, 18, requires_grad=True)

    embedding(waveforms).square().mean().backward()

    assert waveforms.grad is not None
    assert torch.isfinite(waveforms.grad).all()
    assert embedding.projection.weight.grad is not None
