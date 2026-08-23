# Concept: paged KV memory and continuous batching

Each generated token adds a key and value at every attention layer. Different
requests grow at different rates, so their KV cache sizes are dynamic.

## Logical positions and physical blocks

Instead of reserving one large contiguous tensor per sequence, divide physical
KV memory into equal blocks. A sequence owns a logical block table:

```text
logical block:   0  1  2
physical block:  4  1  7
```

The sequence still reads positions in logical order even though physical storage
is non-contiguous. For block size `B`, logical position `t` maps to:

```text
logical_block = t // B
offset        = t % B
physical      = block_table[logical_block]
```

A new physical block is allocated only when the current final block is full.
Unused capacity is therefore bounded by fewer than one block per sequence.

## Free and reuse

When a sequence finishes, return all referenced blocks to the free list. Clearing
their tensors makes reuse visible and prevents stale values from confusing this
teaching simulation.

## Continuous batching

A static batch waits for every row to finish before replacing any row. Continuous
batching operates in iterations:

1. Admit arrived requests into open active slots and prefill their prompts.
2. Decode one token for every active request.
3. Finish requests that emit EOS or reach their token budget.
4. Free their KV blocks immediately.
5. Admit more work before the next decode iteration.

This keeps the active set useful even when request lengths differ.

## Scope

The block tensors and scheduler events expose ownership, fragmentation, and
request state. Real engines must also coordinate model execution, multiple
layers/devices, cache-aware preemption, prefix sharing, and optimized kernels.
This CPU simulation teaches the data structures without imitating those details.
