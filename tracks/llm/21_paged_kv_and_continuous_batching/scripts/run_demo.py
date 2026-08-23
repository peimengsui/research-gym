"""Simulate paged KV allocation while requests continuously enter a batch."""

import shutil
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        KVBlockPool,
        PagedKVCache,
        ServingRequest,
        run_continuous_batching,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        KVBlockPool,
        PagedKVCache,
        ServingRequest,
        run_continuous_batching,
    )


def main() -> None:
    pool = KVBlockPool(num_blocks=12, block_size=2, num_heads=2, head_dim=3)
    cache = PagedKVCache(pool)
    requests = [
        ServingRequest("alpha", [1, 2, 3], max_new_tokens=2, arrival_step=0),
        ServingRequest("beta", [4], max_new_tokens=4, arrival_step=0),
        ServingRequest("gamma", [7, 8], max_new_tokens=2, arrival_step=1),
    ]

    def decode(request: ServingRequest) -> int:
        return request.prompt_token_ids[-1] + len(request.generated_token_ids) + 1

    result = run_continuous_batching(
        requests,
        cache,
        max_active_requests=2,
        decode_next_token=decode,
    )

    print("step  request  action   token")
    for event in result.events:
        token = "-" if event.token_id is None else str(event.token_id)
        print(f"{event.step:>4}  {event.request_id:<7}  {event.action:<7}  {token}")
    print(f"completion order: {list(result.completed_request_ids)}")
    print(f"total scheduler steps: {result.total_steps}")
    print(f"free blocks after completion: {len(pool.free_block_ids)}/{pool.num_blocks}")
    print("Gamma enters while beta is still decoding; finished blocks are reused.")
    print("This is a state/memory simulation, not a production attention kernel.")


if __name__ == "__main__":
    main()
