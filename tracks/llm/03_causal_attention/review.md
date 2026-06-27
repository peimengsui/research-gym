# Review checklist

- Can you explain why the mask is lower-triangular?
- Did you scale attention scores by `sqrt(channels)`?
- Did you apply the mask before the softmax?
- Do attention weights sum to one over allowed key positions?
- Can you explain why changing a future token must not change earlier outputs?
