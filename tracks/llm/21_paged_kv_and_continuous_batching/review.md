# Review questions

- Why do dynamically growing sequences make contiguous reservation wasteful?
- How is a logical token position translated into a physical block and offset?
- Can one sequence's physical block IDs be non-contiguous?
- What bounds internal fragmentation in this block scheme?
- Why should released block contents be cleared in this lesson?
- When can a newly arrived request enter a continuous batch?
- Why is prefill different from a one-token decode iteration?
- What state must be removed immediately when a request reaches EOS?
- What happens when the physical block pool is exhausted?
- Which production concerns are intentionally absent from this simulation?
