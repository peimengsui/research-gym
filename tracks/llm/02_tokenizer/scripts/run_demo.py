"""Train and inspect the learner's tokenizer on a tiny inline corpus."""

from implementation import BPETokenizer


def main() -> None:
    corpus = (
        "research gym makes research implementations understandable. "
        "small research models make ideas inspectable. "
    ) * 8
    tokenizer = BPETokenizer.train(corpus, num_merges=20)

    sample = "research models make ideas understandable."
    token_ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(token_ids)

    print(f"Character count: {len(sample)}")
    print(f"Token count:     {len(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Learned merges:  {len(tokenizer.merges)}")
    print(f"Token IDs:       {token_ids}")
    print(f"Round trip:      {decoded}")
    print(f"Unknown example: {tokenizer.decode(tokenizer.encode('research 🧪'))}")


if __name__ == "__main__":
    main()
