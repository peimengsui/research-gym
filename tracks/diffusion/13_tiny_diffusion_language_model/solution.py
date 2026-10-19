"""Reference solution for a tiny masked diffusion language model."""

import torch
import torch.nn as nn

from provided import (
    UnmaskStep,
    linear_mask_schedule,
    masked_token_cross_entropy,
    q_sample_tokens,
    sinusoidal_timestep_embedding,
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
        sequence_length = input_ids.shape[1]
        positions = torch.arange(sequence_length, device=input_ids.device)
        timestep_features = sinusoidal_timestep_embedding(
            timesteps,
            self.embedding_dim,
        )
        timestep_features = self.timestep_projection(timestep_features)

        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = hidden + timestep_features[:, None, :]

        # No causal attention mask is passed: every valid token may use context
        # on both sides. The key-padding mask only hides padded key/value slots.
        hidden = self.transformer(
            hidden,
            src_key_padding_mask=~attention_mask,
        )
        return self.output_projection(self.final_norm(hidden))


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

    if timesteps is None:
        timesteps = torch.randint(
            1,
            model.num_timesteps,
            (clean_ids.shape[0],),
            device=clean_ids.device,
            generator=generator,
        )
    corruption = q_sample_tokens(
        clean_ids=clean_ids,
        timesteps=timesteps,
        mask_probabilities=linear_mask_schedule(model.num_timesteps),
        attention_mask=attention_mask,
        special_token_mask=special_token_mask,
        mask_token_id=mask_token_id,
        uniforms=uniforms,
        generator=generator,
    )
    logits = model(
        corruption.corrupted_ids,
        corruption.timesteps,
        attention_mask,
    )
    return masked_token_cross_entropy(logits, clean_ids, corruption.loss_mask)


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
    allowed_logits = logits.clone()
    if forbidden_token_ids:
        allowed_logits[..., list(forbidden_token_ids)] = -torch.inf
    probabilities = torch.softmax(allowed_logits, dim=-1)
    confidence, predicted_ids = probabilities.max(dim=-1)

    selected_mask = torch.zeros_like(editable_mask)
    for row in range(logits.shape[0]):
        candidate_positions = torch.where(editable_mask[row])[0]
        count = min(int(reveal_counts[row].item()), candidate_positions.numel())
        if count == 0:
            continue
        candidate_confidence = confidence[row, candidate_positions]
        top_indices = torch.topk(candidate_confidence, k=count).indices
        selected_mask[row, candidate_positions[top_indices]] = True
    return predicted_ids, confidence, selected_mask


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

    logits = model(input_ids, timesteps, attention_mask)
    editable_mask = (input_ids == mask_token_id) & attention_mask & ~special_token_mask
    predicted_ids, confidence, selected_mask = select_confident_predictions(
        logits,
        editable_mask,
        reveal_counts,
        forbidden_token_ids,
    )
    output_ids = torch.where(selected_mask, predicted_ids, input_ids)
    return UnmaskStep(
        output_ids=output_ids,
        selected_mask=selected_mask,
        predicted_ids=predicted_ids,
        confidence=confidence,
    )


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
    current_ids = initial_ids.clone()
    trajectory = [current_ids.clone()]
    reverse_timesteps = list(range(num_timesteps - 1, 0, -1))

    for step_index, timestep in enumerate(reverse_timesteps):
        editable_mask = (
            (current_ids == mask_token_id) & attention_mask & ~special_token_mask
        )
        remaining = editable_mask.sum(dim=1)
        steps_left = len(reverse_timesteps) - step_index
        reveal_counts = torch.div(
            remaining + steps_left - 1,
            steps_left,
            rounding_mode="floor",
        )
        timestep_batch = torch.full(
            (current_ids.shape[0],),
            timestep,
            dtype=torch.long,
            device=current_ids.device,
        )
        update = unmask_step(
            model,
            current_ids,
            timestep_batch,
            attention_mask,
            special_token_mask,
            mask_token_id,
            reveal_counts,
            forbidden_token_ids,
        )
        current_ids = update.output_ids
        trajectory.append(current_ids.clone())

    return current_ids, torch.stack(trajectory)
