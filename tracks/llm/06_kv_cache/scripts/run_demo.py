"""Compare full-context greedy decoding with KV-cache decoding."""

import time

import torch

from implementation import TinyCachedGPT


def main() -> None:
    torch.manual_seed(7)

    model = TinyCachedGPT(vocab_size=16, block_size=24, embed_dim=24, num_layers=2)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    max_new_tokens = 12

    start = time.perf_counter()
    full = model.generate_full(prompt, max_new_tokens=max_new_tokens)
    full_seconds = time.perf_counter() - start

    start = time.perf_counter()
    cached = model.generate_cached(prompt, max_new_tokens=max_new_tokens)
    cached_seconds = time.perf_counter() - start

    print(f"Prompt:                 {prompt.tolist()[0]}")
    print(f"Full-context generated: {full.tolist()[0]}")
    print(f"Cached generated:       {cached.tolist()[0]}")
    print(f"Sequences match:        {torch.equal(full, cached)}")
    print(f"Full-context time:      {full_seconds:.6f}s")
    print(f"Cached time:            {cached_seconds:.6f}s")
    print("Note: tiny CPU timings are noisy; equality is the important check here.")


if __name__ == "__main__":
    main()
