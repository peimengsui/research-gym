import torch
import torch.nn as nn

from implementation import (
    TinyDiscreteDenoiser,
    discrete_denoising_loss,
    iterative_unmask,
    select_confident_predictions,
    unmask_step,
)
from provided import make_toy_token_batch


class PositionOracle(nn.Module):
    """Return deterministic per-position targets with decreasing confidence."""

    def __init__(self, target_ids: torch.Tensor, vocabulary_size: int) -> None:
        super().__init__()
        self.register_buffer("target_ids", target_ids)
        self.vocabulary_size = vocabulary_size

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps, attention_mask
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocabulary_size,
            device=input_ids.device,
        )
        targets = self.target_ids.expand(batch_size, -1)
        strengths = torch.arange(
            sequence_length,
            0,
            -1,
            dtype=logits.dtype,
            device=logits.device,
        )
        logits.scatter_(2, targets[..., None], strengths[None, :, None])
        return logits


def make_model() -> TinyDiscreteDenoiser:
    torch.manual_seed(0)
    return TinyDiscreteDenoiser(
        vocabulary_size=11,
        max_length=6,
        num_timesteps=5,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
    )


def test_denoiser_returns_token_logits_and_uses_right_context() -> None:
    model = make_model()
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    timesteps = torch.tensor([3])
    first = torch.tensor([[1, 3, 4, 2]])
    second = torch.tensor([[1, 3, 8, 2]])

    first_logits = model(first, timesteps, attention_mask)
    second_logits = model(second, timesteps, attention_mask)

    assert first_logits.shape == (1, 4, 11)
    assert not torch.allclose(first_logits[:, 1], second_logits[:, 1])


def test_padding_tokens_do_not_change_valid_position_logits() -> None:
    model = make_model()
    attention_mask = torch.tensor([[True, True, True, True, False, False]])
    timesteps = torch.tensor([2])
    first = torch.tensor([[1, 3, 4, 2, 0, 0]])
    second = torch.tensor([[1, 3, 4, 2, 8, 9]])

    first_logits = model(first, timesteps, attention_mask)
    second_logits = model(second, timesteps, attention_mask)

    assert torch.allclose(first_logits[:, :4], second_logits[:, :4], atol=1e-6)


def test_denoising_loss_is_finite_and_backpropagates() -> None:
    model = make_model()
    batch = make_toy_token_batch()
    timesteps = torch.full((batch.input_ids.shape[0],), 4, dtype=torch.long)

    loss = discrete_denoising_loss(
        model,
        batch.input_ids,
        batch.attention_mask,
        batch.special_token_mask,
        batch.mask_token_id,
        timesteps=timesteps,
        uniforms=torch.zeros_like(batch.input_ids, dtype=torch.float),
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.grad.abs().sum() > 0


def test_confidence_selection_filters_tokens_and_respects_each_row_count() -> None:
    logits = torch.zeros(2, 3, 6)
    logits[..., 5] = 20.0  # Forbidden [MASK] would otherwise always win.
    logits[0, 0, 3] = 9.0
    logits[0, 1, 4] = 7.0
    logits[1, 0, 4] = 8.0
    logits[1, 2, 3] = 6.0
    editable = torch.tensor([[True, True, False], [True, False, True]])

    predicted, confidence, selected = select_confident_predictions(
        logits,
        editable,
        torch.tensor([1, 2]),
        forbidden_token_ids=(0, 1, 2, 5),
    )

    assert predicted[0, 0].item() == 3
    assert predicted[1, 0].item() == 4
    assert confidence.shape == editable.shape
    assert selected.tolist() == [[True, False, False], [True, False, True]]


def test_unmask_step_preserves_special_padding_and_fixed_tokens() -> None:
    input_ids = torch.tensor([[1, 10, 7, 10, 2, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, False]])
    special_mask = torch.tensor([[True, False, False, False, True, True]])
    oracle = PositionOracle(torch.tensor([[4, 3, 4, 5, 4, 4]]), 11)

    update = unmask_step(
        oracle,
        input_ids,
        torch.tensor([3]),
        attention_mask,
        special_mask,
        mask_token_id=10,
        reveal_counts=torch.tensor([2]),
        forbidden_token_ids=(0, 1, 2, 10),
    )

    assert update.output_ids.tolist() == [[1, 3, 7, 5, 2, 0]]
    assert update.selected_mask.tolist() == [[False, True, False, True, False, False]]


def test_iterative_unmask_fills_all_slots_monotonically() -> None:
    initial_ids = torch.tensor([[1, 10, 7, 10, 2, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, False]])
    special_mask = torch.tensor([[True, False, False, False, True, True]])
    oracle = PositionOracle(torch.tensor([[4, 3, 4, 5, 4, 4]]), 11)

    final_ids, trajectory = iterative_unmask(
        oracle,
        initial_ids,
        attention_mask,
        special_mask,
        mask_token_id=10,
        num_timesteps=4,
        forbidden_token_ids=(0, 1, 2, 10),
    )

    mask_counts = (trajectory == 10).sum(dim=2).squeeze(1)
    assert trajectory.shape == (4, 1, 6)
    assert torch.all(mask_counts[1:] <= mask_counts[:-1])
    assert final_ids.tolist() == [[1, 3, 7, 5, 2, 0]]
    assert mask_counts[-1].item() == 0


def test_iterative_unmask_leaves_a_complete_template_unchanged() -> None:
    initial_ids = torch.tensor([[1, 3, 7, 5, 2, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, False]])
    special_mask = torch.tensor([[True, False, False, False, True, True]])
    oracle = PositionOracle(torch.tensor([[4, 9, 9, 9, 4, 4]]), 11)

    final_ids, trajectory = iterative_unmask(
        oracle,
        initial_ids,
        attention_mask,
        special_mask,
        mask_token_id=10,
        num_timesteps=3,
        forbidden_token_ids=(0, 1, 2, 10),
    )

    assert torch.equal(final_ids, initial_ids)
    assert torch.equal(trajectory, initial_ids.expand(3, -1, -1))
