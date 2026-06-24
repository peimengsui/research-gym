# Tokenizer Fundamentals

Build a small byte-pair-style tokenizer that begins with individual characters
and learns frequent adjacent-token merges.

You will implement:

- adjacent pair counting
- deterministic merge selection
- non-overlapping pair replacement
- text encoding and decoding
- unknown-character handling

Start the lesson from the repository root:

```bash
uv run rgym start llm.02_tokenizer
cd workspace/llm.02_tokenizer
uv run rgym test
```

Read `concept.md` and `guide.md`, then edit only `implementation.py`.

After completing the exercise, compare your implementation with
`tracks/llm/02_tokenizer/solution.py` in the source repository.
