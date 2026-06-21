# Bigram Language Model

Build the smallest useful next-token language model: a table that maps every
token directly to logits for the token that should follow it.

You will implement:

- a token embedding table
- a forward pass with optional cross-entropy loss
- autoregressive sampling

Start the lesson from the repository root:

```bash
uv run rgym start llm.01_bigram_lm
cd workspace/llm.01_bigram_lm
uv run rgym test
```

Read `concept.md` for the idea and `guide.md` for the implementation sequence.
Edit only `implementation.py` in your workspace.

After completing the exercise, you can compare your code with the source
reference at `tracks/llm/01_bigram_lm/solution.py`.
