# Review questions

- Why must prompt tokens be removed before exact-match evaluation?
- Should EOS be included in a human-readable reference answer?
- Why is the first candidate token scored from the final prompt logit?
- What length preference results from summing candidate log probabilities?
- When might sum, mean, or another normalization be preferable?
- Why can exact match reject a semantically correct answer?
- What information in an evaluation record helps diagnose a failure?
- Why should an evaluation helper restore the model's original mode?
- How would padded prompts or batched variable-length candidates change this code?
- What larger or less synthetic evaluation data would you add next?
