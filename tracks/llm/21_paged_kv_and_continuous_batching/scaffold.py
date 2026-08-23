"""Learner scaffold for paged KV memory and continuous batching.

Synthetic token-to-KV conversion is complete in provided.py. TODOs focus on
physical block management, logical block tables, and request scheduling.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from provided import token_to_kv


class KVBlockPool:
    """Own fixed-size physical key/value blocks and a free list."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
    ):
        if min(num_blocks, block_size, num_heads, head_dim) <= 0:
            raise ValueError("pool dimensions must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        shape = (num_blocks, block_size, num_heads, head_dim)
        self.keys = torch.zeros(shape)
        self.values = torch.zeros(shape)
        self.free_block_ids = list(range(num_blocks))
        self.allocated_block_ids: set[int] = set()

    def allocate(self) -> int:
        # TODO 1: Raise RuntimeError when no free block remains. Otherwise remove
        # the first free ID, mark it allocated, and return it.
        raise NotImplementedError

    def release(self, block_id: int) -> None:
        # TODO 2: Require an allocated ID, remove it from the allocated set, zero
        # both physical blocks, return the ID to the free list, and sort that list
        # so allocation remains deterministic.
        raise NotImplementedError

    def write(
        self,
        block_id: int,
        offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if block_id not in self.allocated_block_ids:
            raise ValueError("block must be allocated before writing")
        if not 0 <= offset < self.block_size:
            raise ValueError("offset is outside the block")
        expected = (self.num_heads, self.head_dim)
        if key.shape != expected or value.shape != expected:
            raise ValueError(f"key and value must have shape {expected}")
        self.keys[block_id, offset] = key
        self.values[block_id, offset] = value


@dataclass
class SequenceBlockTable:
    block_ids: list[int] = field(default_factory=list)
    length: int = 0


class PagedKVCache:
    """Map logical sequence positions onto non-contiguous physical blocks."""

    def __init__(self, pool: KVBlockPool):
        self.pool = pool
        self.tables: dict[str, SequenceBlockTable] = {}

    def start_sequence(self, sequence_id: str) -> None:
        if not sequence_id or sequence_id in self.tables:
            raise ValueError("sequence_id must be new and non-empty")
        self.tables[sequence_id] = SequenceBlockTable()

    def append(
        self,
        sequence_id: str,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if sequence_id not in self.tables:
            raise ValueError("sequence must be started before append")
        # TODO 3: Allocate a new physical block whenever length % block_size == 0.
        # Translate the current logical position into logical block index and
        # in-block offset, look up its physical block ID, write K/V, then increment
        # sequence length.
        raise NotImplementedError

    def read(self, sequence_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_id not in self.tables:
            raise ValueError("unknown sequence_id")
        # TODO 4: Visit physical block IDs in logical order. Read only valid rows
        # from the final partially filled block and concatenate into tensors shaped
        # (sequence_length, num_heads, head_dim). Return matching empty tensors for
        # a zero-length sequence.
        raise NotImplementedError

    def free_sequence(self, sequence_id: str) -> None:
        if sequence_id not in self.tables:
            raise ValueError("unknown sequence_id")
        # TODO 5: Remove the table and release every physical block it references.
        raise NotImplementedError

    def internal_fragmentation_slots(self) -> int:
        # TODO 6: For each active sequence, subtract used length from
        # len(block_ids) * block_size and sum the unused reserved slots.
        raise NotImplementedError


@dataclass
class ServingRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    arrival_step: int = 0
    generated_token_ids: list[int] = field(default_factory=list)
    state: str = "waiting"


@dataclass(frozen=True)
class SchedulerEvent:
    step: int
    request_id: str
    action: str
    token_id: int | None = None


@dataclass(frozen=True)
class SchedulerResult:
    events: tuple[SchedulerEvent, ...]
    total_steps: int
    completed_request_ids: tuple[str, ...]


def _append_token(cache: PagedKVCache, request_id: str, token_id: int) -> None:
    key, value = token_to_kv(token_id, cache.pool.num_heads, cache.pool.head_dim)
    cache.append(request_id, key, value)


def run_continuous_batching(
    requests: Sequence[ServingRequest],
    cache: PagedKVCache,
    max_active_requests: int,
    decode_next_token: Callable[[ServingRequest], int],
    eos_token_id: int | None = None,
) -> SchedulerResult:
    """Admit arrived requests and decode one token per active request per step."""

    if max_active_requests <= 0:
        raise ValueError("max_active_requests must be positive")
    request_ids = [request.request_id for request in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request IDs must be unique")
    for request in requests:
        if (
            not request.request_id
            or not request.prompt_token_ids
            or request.max_new_tokens <= 0
            or request.arrival_step < 0
            or request.state != "waiting"
        ):
            raise ValueError("requests must be valid and initially waiting")

    # TODO 7:
    # 1. Stable-sort pending requests by (arrival_step, input order).
    # 2. At each step, jump over idle time, then admit arrived work until the
    #    active limit. Start its cache, append prompt KV, set active, and log prefill.
    # 3. Decode one token for every active request, append KV, and log decode.
    # 4. On EOS or max_new_tokens, mark finished, free its cache immediately,
    #    remove it from active work, and log finish.
    # 5. Return immutable events, total scheduler steps, and completion order.
    raise NotImplementedError
