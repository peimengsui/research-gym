# Review

After your tests pass, explain:

- why one tubelet can contain motion information while one image patch cannot
- how the flat token index maps to time, patch row, and patch column
- why reconstruction is a useful test even though a model only needs extraction
- what separate temporal and spatial position tables encode
- what assumptions fixed-shape batched clips make

Then compare your implementation with `solution.locked.py`. Focus on tensor
ordering and validation rather than matching syntax.
