"""Reference solution for modern Gated Linear Attention (GLA)."""

import torch
import torch.nn.functional as F
from torch import nn

from provided import (
    GatedLinearAttentionState,
    merge_heads,
    rms_norm_per_head,
    split_heads,
    validate_gate_logits,
    validate_gla_inputs,
    validate_module_configuration,
    validate_module_forward,
    validate_state,
)


def gate_log_decay(
    gate_logits: torch.Tensor,
    normalizer: float = 16.0,
) -> torch.Tensor:
    """Convert arbitrary logits to stable non-positive log retention values."""

    validate_gate_logits(gate_logits, normalizer)
    return F.logsigmoid(gate_logits) / normalizer


def gated_linear_attention_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    state: GatedLinearAttentionState | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, GatedLinearAttentionState]:
    """Run the GLA recurrence over one sequence chunk."""

    dimensions = validate_gla_inputs(query, key, value, log_decay, scale)
    batch_size, num_heads, _, key_dim, value_dim, resolved_scale = dimensions
    if state is None:
        memory = query.new_zeros(batch_size, num_heads, key_dim, value_dim)
    else:
        validate_state(
            state,
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            query,
        )
        memory = state.memory

    outputs = []
    for position in range(query.shape[2]):
        decay = log_decay[:, :, position].exp().unsqueeze(-1)
        current_key = key[:, :, position]
        current_value = value[:, :, position]
        memory = decay * memory
        memory = memory + torch.einsum(
            "bhk,bhv->bhkv",
            current_key,
            current_value,
        )
        current_output = torch.einsum(
            "bhk,bhkv->bhv",
            query[:, :, position],
            memory,
        )
        outputs.append(current_output * resolved_scale)

    output = torch.stack(outputs, dim=2)
    return output, GatedLinearAttentionState(memory)


def gated_linear_attention_parallel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    state: GatedLinearAttentionState | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, GatedLinearAttentionState]:
    """Run an explicit parallel expansion of the same gated recurrence."""

    dimensions = validate_gla_inputs(query, key, value, log_decay, scale)
    batch_size, num_heads, sequence_length, key_dim, value_dim, resolved_scale = (
        dimensions
    )
    if state is not None:
        validate_state(
            state,
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            query,
        )

    cumulative_log_decay = log_decay.cumsum(dim=2)
    pairwise_log_decay = cumulative_log_decay.unsqueeze(3) - (
        cumulative_log_decay.unsqueeze(2)
    )
    causal_mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=query.device,
    ).tril()
    pairwise_log_decay = pairwise_log_decay.masked_fill(
        ~causal_mask[None, None, :, :, None],
        float("-inf"),
    )
    retention = pairwise_log_decay.exp()
    weights = torch.einsum(
        "bhtk,bhsk,bhtsk->bhts",
        query,
        key,
        retention,
    )
    output = torch.einsum("bhts,bhsv->bhtv", weights, value) * resolved_scale

    if state is not None:
        incoming_retention = cumulative_log_decay.exp()
        decayed_memory = incoming_retention.unsqueeze(-1) * state.memory.unsqueeze(2)
        output = output + (
            torch.einsum("bhtk,bhtkv->bhtv", query, decayed_memory) * resolved_scale
        )

    final_retention = retention[:, :, -1]
    final_memory = torch.einsum(
        "bhsk,bhsv,bhsk->bhkv",
        key,
        value,
        final_retention,
    )
    if state is not None:
        final_memory = final_memory + (
            cumulative_log_decay[:, :, -1].exp().unsqueeze(-1) * state.memory
        )
    return output, GatedLinearAttentionState(final_memory)


class GatedLinearAttention(nn.Module):
    """Educational GLA layer with low-rank forget and SiLU output gates."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        gate_rank: int = 4,
        gate_logit_normalizer: float = 16.0,
        norm_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.head_dim = validate_module_configuration(
            embed_dim,
            num_heads,
            gate_rank,
            gate_logit_normalizer,
            norm_epsilon,
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gate_logit_normalizer = gate_logit_normalizer
        self.norm_epsilon = norm_epsilon
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate_down = nn.Linear(embed_dim, gate_rank, bias=False)
        self.gate_up = nn.Linear(gate_rank, embed_dim, bias=True)
        self.output_gate = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: GatedLinearAttentionState | None = None,
        use_recurrent: bool = False,
    ) -> tuple[torch.Tensor, GatedLinearAttentionState]:
        """Run parallel or recurrent GLA and return fixed-size matrix memory."""

        validate_module_forward(
            x,
            state,
            self.embed_dim,
            self.num_heads,
            self.head_dim,
            use_recurrent,
        )
        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        gate_logits = split_heads(
            self.gate_up(self.gate_down(x)),
            self.num_heads,
        )
        log_decay = gate_log_decay(gate_logits, self.gate_logit_normalizer)

        attention_function = (
            gated_linear_attention_recurrent
            if use_recurrent
            else gated_linear_attention_parallel
        )
        attended, state = attention_function(
            query,
            key,
            value,
            log_decay,
            state,
        )
        attended = rms_norm_per_head(attended, self.norm_epsilon)
        output_gate = F.silu(split_heads(self.output_gate(x), self.num_heads))
        attended = attended * output_gate
        return self.output(merge_heads(attended)), state
