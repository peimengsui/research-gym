# KV Blocks and Request Scheduling

Serving requests have different prompt lengths, arrival times, and stopping
points. Reserving one maximum-size contiguous KV cache per request wastes memory,
while waiting for a fixed batch to finish leaves newly available capacity idle.

You will implement:

- allocation and reuse of fixed-size physical KV blocks
- logical sequence block tables
- append, read, free, and fragmentation accounting
- request prefill and one-token decode states
- continuous admission between decoding iterations
- immediate KV release at EOS or a generation limit

`provided.py` turns token IDs into deterministic synthetic keys and values. The
lesson tests memory mapping and scheduler behavior on CPU; it does not implement
a fused attention kernel or make production throughput claims.

## Start

```bash
uv run rgym start llm.21_paged_kv_and_continuous_batching
cd workspace/llm.21_paged_kv_and_continuous_batching
uv run rgym test
uv run rgym run
```
