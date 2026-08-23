# Guide

Open `implementation.py`. Complete the seven TODOs in order.

## 1–2. Allocate and release physical blocks

Allocation removes the lowest free ID and marks it allocated. Release validates
ownership, clears keys and values, and returns the ID to a sorted free list.

## 3. Append through a block table

Allocate when sequence length is a multiple of block size. Convert the current
logical position into a logical block index and offset, look up the physical ID,
write the supplied `(num_heads, head_dim)` tensors, and increment length.

## 4. Read logical order

Visit physical IDs in block-table order. All non-final blocks are full; the final
block may be partial. Concatenate only valid slots into
`(sequence_length, num_heads, head_dim)`.

## 5–6. Free and measure fragmentation

Remove a sequence table and release all its blocks. For active sequences,
internal fragmentation is allocated slots minus used positions.

## 7. Schedule continuously

Stable-sort by arrival and input order. At each scheduler step:

1. Jump to the next arrival if no work is active.
2. Admit arrived requests until the active limit.
3. Prefill each admitted prompt into its paged cache.
4. Decode one token for every active request.
5. Finish and free requests immediately at EOS or their token limit.

Log prefill, decode, and finish events so scheduling decisions remain inspectable.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include treating physical IDs as logical order, allocating on every
token, reading unused final-block rows, forgetting to free at EOS, mutating the
active list without iterating over a copy, or admitting new work only after every
older request has completed.
