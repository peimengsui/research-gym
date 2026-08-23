# Hints

## Hint 1

Use `pop(0)` for deterministic allocation and sort the free list after release.

## Hint 2

Allocate when `table.length % pool.block_size == 0`.

## Hint 3

The current physical block is
`table.block_ids[table.length // pool.block_size]` immediately after allocation.

## Hint 4

While reading, track `remaining` and take
`min(pool.block_size, remaining)` rows from each physical block.

## Hint 5

Iterate decoding over `list(active)` because finishing removes requests from the
original active list.

## Hint 6

When no request is active, set the scheduler step to the next pending arrival to
avoid emitting meaningless idle iterations.
