# Review

After your tests pass, explain:

- which tokens can communicate during the spatial pass
- which tokens can communicate during the temporal pass
- why two passes can connect arbitrary spatiotemporal positions
- how the attention work compares with full video attention
- why video encoding is bidirectional while generated text remains causal

Compare your implementation with `solution.locked.py`, especially the reshape
and permutation sequence around temporal attention.
