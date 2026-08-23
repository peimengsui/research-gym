# Review

After your tests pass, explain:

- why video-prefix attention is bidirectional while text attention is causal
- where logits for the first answer token come from
- why finished rows receive EOS during rectangular batched generation
- what compute this lesson repeats at every generation step
- why exact-match generation and candidate ranking measure different behavior
- which pieces came directly from the image-language lessons

Compare with `solution.locked.py`. Pay special attention to mask boundaries and
the candidate-scoring offset.
