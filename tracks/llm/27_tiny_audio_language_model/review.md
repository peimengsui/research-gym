# Review

After the tests pass, explain:

- why padded audio slots remain in the sequence
- which prefix positions a text query can attend to
- why invalid audio query rows are all false
- why generation and evaluation do not need to be rewritten per modality
- where the first answer token is predicted
- how KV caching could avoid repeated audio encoding

Compare the three TODO implementations with `solution.locked.py` and inspect the
provided generation/evaluation code as carried-forward reference material.
