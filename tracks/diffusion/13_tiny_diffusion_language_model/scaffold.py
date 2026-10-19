"""Learner scaffold for a tiny masked diffusion language model."""

import torch
import torch.nn as nn

from provided import (
    UnmaskStep,
    linear_mask_schedule,  # noqa: F401 - useful for TODO 2
    masked_token_cross_entropy,  # noqa: F401 - useful for TODO 2
    q_sample_tokens,  # noqa: F401 - useful for TODO 2
    sinusoidal_timestep_embedding,  # noqa: F401 - useful for TODO 1
    validate_generation_inputs,
    validate_model_inputs,
    validate_selection_inputs,
)


class TinyDiscreteDenoiser(nn.Module):
    """A bidirectional Transformer that predicts clean tokens at every position."""

    def __init__(
        self,
        vocabulary_size: int,
        max_length: int,
        num_timesteps: int,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2")
        if max_length < 1 or num_timesteps < 2:
            raise ValueError("max_length must be positive and num_timesteps at least 2")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.vocabulary_size = vocabulary_size
        self.max_length = max_length
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.timestep_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocabulary_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return clean-token logits with shape (batch, sequence, vocabulary)."""

        validate_model_inputs(
            input_ids,
            timesteps,
            attention_mask,
            self.vocabulary_size,
            self.max_length,
            self.num_timesteps,
        )
        # TODO 1: Add token, position, and projected sinusoidal timestep
        # embeddings. Run the Transformer with a key-padding mask and return
        # vocabulary logits. Do not pass a causal mask: this denoiser must use
        # valid context on both sides of a masked token.
        raise NotImplementedError


def discrete_denoising_loss(
    model: TinyDiscreteDenoiser,
    clean_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
    timesteps: torch.Tensor | None = None,
    uniforms: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Corrupt clean tokens and score only positions replaced by `[MASK]`."""

    # TODO 2: Sample nonzero timesteps when none are supplied, call the
    # provided q_sample_tokens, run the model on corrupted_ids, and use the
    # provided masked_token_cross_entropy with corruption.loss_mask.
    raise NotImplementedError


def select_confident_predictions(
    logits: torch.Tensor,
    editable_mask: torch.Tensor,
    reveal_counts: torch.Tensor,
    forbidden_token_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict ordinary tokens and select each row's most confident positions."""

    validate_selection_inputs(
        logits,
        editable_mask,
        reveal_counts,
        forbidden_token_ids,
    )
    # TODO 3: Before softmax, set each forbidden token's logit to -infinity.
    # Find every position's best allowed token and its probability. For each
    # batch row, select at most reveal_counts[row] editable positions with the
    # highest confidence. Return (predicted_ids, confidence, selected_mask).
    # A small per-row loop is clearer than forcing variable top-k into one op.
    raise NotImplementedError


def unmask_step(
    model: TinyDiscreteDenoiser,
    input_ids: torch.Tensor,
    timesteps: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
    reveal_counts: torch.Tensor,
    forbidden_token_ids: tuple[int, ...],
) -> UnmaskStep:
    """Reveal selected masks while preserving every non-editable position."""

    # TODO 4: Editable means currently [MASK], valid according to
    # attention_mask, and not special. Predict and select tokens, then replace
    # only selected positions with torch.where. Already-fixed prompt/content
    # tokens must remain unchanged even if the model prefers another token.
    raise NotImplementedError


@torch.no_grad()
def iterative_unmask(
    model: TinyDiscreteDenoiser,
    initial_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
    num_timesteps: int,
    forbidden_token_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate content through monotonic confidence-based unmasking.

    Returns the final token IDs and a trajectory with shape
    (num_timesteps, batch, sequence), including the initial state.
    """

    validate_generation_inputs(
        initial_ids,
        attention_mask,
        special_token_mask,
        mask_token_id,
        num_timesteps,
        forbidden_token_ids,
    )
    # TODO 5: Iterate timesteps T-1, ..., 1. At each step count remaining
    # editable masks and reveal ceil(remaining / steps_left) per row. Append
    # every state, including the initial state, and return the final IDs plus
    # torch.stack(trajectory). This schedule guarantees all editable masks are
    # filled on the last step while allowing confident positions to go first.
    raise NotImplementedError
