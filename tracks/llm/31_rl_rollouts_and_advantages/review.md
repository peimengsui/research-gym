# Review checklist

- Is `response_mask` aligned with labels rather than raw model inputs?
- Is the first response token included and every prompt token excluded?
- Are padded response positions excluded?
- Are token log-probabilities retained individually instead of summed?
- Are old-policy and reference log-probabilities frozen snapshots?
- Is token KL defined as `old_logprob - reference_logprob`?
- Are KL penalties applied only to response tokens?
- Is the sequence score added only at the final valid response token?
- Does final-token GAE bootstrap from zero?
- Does GAE stop across every masked boundary?
- Are returns defined as `advantages + values` at valid positions?
- Are prompt and padding statistics all zero?
- Are rollout tensors detached from autograd?
- Can the resulting batch be reused across multiple PPO epochs?
