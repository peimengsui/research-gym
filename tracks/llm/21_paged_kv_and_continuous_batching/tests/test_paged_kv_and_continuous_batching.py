import pytest
import torch

from implementation import (
    KVBlockPool,
    PagedKVCache,
    ServingRequest,
    run_continuous_batching,
)
from provided import token_to_kv


def make_cache(num_blocks: int = 8, block_size: int = 2) -> PagedKVCache:
    return PagedKVCache(
        KVBlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=2,
            head_dim=3,
        )
    )


def append_token(cache: PagedKVCache, sequence_id: str, token_id: int) -> None:
    key, value = token_to_kv(token_id, cache.pool.num_heads, cache.pool.head_dim)
    cache.append(sequence_id, key, value)


def test_block_pool_allocates_releases_and_reuses_lowest_id() -> None:
    pool = KVBlockPool(2, 2, 1, 2)

    first = pool.allocate()
    second = pool.allocate()
    pool.write(first, 0, torch.ones(1, 2), -torch.ones(1, 2))
    pool.release(first)
    reused = pool.allocate()

    assert (first, second, reused) == (0, 1, 0)
    assert torch.equal(pool.keys[reused], torch.zeros_like(pool.keys[reused]))


def test_block_pool_reports_exhaustion() -> None:
    pool = KVBlockPool(1, 2, 1, 1)
    pool.allocate()

    with pytest.raises(RuntimeError, match="exhausted"):
        pool.allocate()


def test_paged_cache_appends_across_blocks_and_reads_logical_order() -> None:
    cache = make_cache(block_size=2)
    cache.start_sequence("a")
    for token_id in [2, 4, 6, 8, 10]:
        append_token(cache, "a", token_id)

    keys, values = cache.read("a")

    assert cache.tables["a"].block_ids == [0, 1, 2]
    assert cache.tables["a"].length == 5
    assert keys.shape == (5, 2, 3)
    assert values.shape == keys.shape
    assert keys[:, 0, 0].tolist() == [2, 4, 6, 8, 10]
    assert values[:, 0, 0].tolist() == [-2, -4, -6, -8, -10]
    assert cache.internal_fragmentation_slots() == 1


def test_logical_sequence_can_use_non_contiguous_physical_blocks() -> None:
    cache = make_cache(block_size=2)
    cache.start_sequence("a")
    cache.start_sequence("b")
    for token in [1, 2]:
        append_token(cache, "a", token)
    for token in [7, 8]:
        append_token(cache, "b", token)
    append_token(cache, "a", 3)

    keys, _ = cache.read("a")

    assert cache.tables["a"].block_ids == [0, 2]
    assert cache.tables["b"].block_ids == [1]
    assert keys[:, 0, 0].tolist() == [1, 2, 3]


def test_free_sequence_returns_all_blocks_to_pool() -> None:
    cache = make_cache(num_blocks=4, block_size=2)
    cache.start_sequence("a")
    for token in [1, 2, 3]:
        append_token(cache, "a", token)

    cache.free_sequence("a")

    assert "a" not in cache.tables
    assert cache.pool.free_block_ids == [0, 1, 2, 3]
    assert not cache.pool.allocated_block_ids


def test_zero_length_sequence_reads_empty_cache() -> None:
    cache = make_cache()
    cache.start_sequence("empty")

    keys, values = cache.read("empty")

    assert keys.shape == (0, 2, 3)
    assert values.shape == keys.shape


def test_continuous_batching_admits_new_request_before_older_work_finishes() -> None:
    cache = make_cache(num_blocks=12, block_size=2)
    requests = [
        ServingRequest("a", [1, 2], max_new_tokens=2, arrival_step=0),
        ServingRequest("b", [3], max_new_tokens=3, arrival_step=0),
        ServingRequest("c", [4], max_new_tokens=1, arrival_step=1),
    ]

    def decode(request: ServingRequest) -> int:
        return request.prompt_token_ids[-1] + len(request.generated_token_ids) + 1

    result = run_continuous_batching(requests, cache, 2, decode)

    c_prefill = next(
        event
        for event in result.events
        if event.request_id == "c" and event.action == "prefill"
    )
    b_finish = next(
        event
        for event in result.events
        if event.request_id == "b" and event.action == "finish"
    )
    assert c_prefill.step == 2
    assert c_prefill.step == b_finish.step
    assert result.completed_request_ids == ("a", "b", "c")
    assert [request.state for request in requests] == ["finished"] * 3
    assert requests[0].generated_token_ids == [3, 4]
    assert requests[1].generated_token_ids == [4, 5, 6]
    assert requests[2].generated_token_ids == [5]
    assert not cache.tables
    assert len(cache.pool.free_block_ids) == cache.pool.num_blocks


def test_scheduler_stops_request_early_at_eos() -> None:
    cache = make_cache()
    request = ServingRequest("eos", [1], max_new_tokens=5)

    result = run_continuous_batching([request], cache, 1, lambda _: 9, eos_token_id=9)

    assert request.generated_token_ids == [9]
    assert request.state == "finished"
    assert result.total_steps == 1
    assert [event.action for event in result.events] == [
        "prefill",
        "decode",
        "finish",
    ]


def test_scheduler_jumps_to_future_arrival_when_idle() -> None:
    cache = make_cache()
    request = ServingRequest("late", [1], max_new_tokens=1, arrival_step=4)

    result = run_continuous_batching([request], cache, 1, lambda _: 2)

    assert result.events[0].step == 4
    assert result.total_steps == 5


def test_scheduler_rejects_duplicate_request_ids() -> None:
    requests = [
        ServingRequest("same", [1], 1),
        ServingRequest("same", [2], 1),
    ]

    with pytest.raises(ValueError, match="unique"):
        run_continuous_batching(requests, make_cache(), 2, lambda _: 0)
