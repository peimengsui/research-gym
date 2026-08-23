"""Reference solution for paged KV memory and continuous batching."""

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
        if not self.free_block_ids:
            raise RuntimeError("KV block pool is exhausted")
        block_id = self.free_block_ids.pop(0)
        self.allocated_block_ids.add(block_id)
        return block_id

    def release(self, block_id: int) -> None:
        if block_id not in self.allocated_block_ids:
            raise ValueError("cannot release a block that is not allocated")
        self.allocated_block_ids.remove(block_id)
        self.keys[block_id].zero_()
        self.values[block_id].zero_()
        self.free_block_ids.append(block_id)
        self.free_block_ids.sort()

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
        table = self.tables[sequence_id]
        if table.length % self.pool.block_size == 0:
            table.block_ids.append(self.pool.allocate())
        logical_block = table.length // self.pool.block_size
        offset = table.length % self.pool.block_size
        physical_block = table.block_ids[logical_block]
        self.pool.write(physical_block, offset, key, value)
        table.length += 1

    def read(self, sequence_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_id not in self.tables:
            raise ValueError("unknown sequence_id")
        table = self.tables[sequence_id]
        key_chunks = []
        value_chunks = []
        remaining = table.length
        for block_id in table.block_ids:
            count = min(self.pool.block_size, remaining)
            key_chunks.append(self.pool.keys[block_id, :count])
            value_chunks.append(self.pool.values[block_id, :count])
            remaining -= count
        if not key_chunks:
            shape = (0, self.pool.num_heads, self.pool.head_dim)
            return torch.empty(shape), torch.empty(shape)
        return torch.cat(key_chunks), torch.cat(value_chunks)

    def free_sequence(self, sequence_id: str) -> None:
        if sequence_id not in self.tables:
            raise ValueError("unknown sequence_id")
        table = self.tables.pop(sequence_id)
        for block_id in table.block_ids:
            self.pool.release(block_id)

    def internal_fragmentation_slots(self) -> int:
        return sum(
            len(table.block_ids) * self.pool.block_size - table.length
            for table in self.tables.values()
        )


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

    indexed = list(enumerate(requests))
    pending = [
        request
        for _, request in sorted(
            indexed, key=lambda item: (item[1].arrival_step, item[0])
        )
    ]
    active: list[ServingRequest] = []
    completed = []
    events = []
    step = 0
    while pending or active:
        if not active and pending and pending[0].arrival_step > step:
            step = pending[0].arrival_step

        while (
            pending
            and pending[0].arrival_step <= step
            and len(active) < max_active_requests
        ):
            request = pending.pop(0)
            cache.start_sequence(request.request_id)
            for token_id in request.prompt_token_ids:
                _append_token(cache, request.request_id, token_id)
            request.state = "active"
            active.append(request)
            events.append(SchedulerEvent(step, request.request_id, "prefill"))

        if not active:
            continue

        for request in list(active):
            token_id = decode_next_token(request)
            _append_token(cache, request.request_id, token_id)
            request.generated_token_ids.append(token_id)
            events.append(SchedulerEvent(step, request.request_id, "decode", token_id))
            reached_limit = len(request.generated_token_ids) >= request.max_new_tokens
            reached_eos = eos_token_id is not None and token_id == eos_token_id
            if reached_limit or reached_eos:
                request.state = "finished"
                cache.free_sequence(request.request_id)
                active.remove(request)
                completed.append(request.request_id)
                events.append(SchedulerEvent(step, request.request_id, "finish"))
        step += 1

    return SchedulerResult(tuple(events), step, tuple(completed))
