# Review questions

- Why does DPO compare the policy against a frozen reference model?
- Which tokens should count in `log p(completion | prompt)`?
- What happens to the loss when the policy already strongly prefers the chosen
  completion?
- Why is DPO simpler to implement than a reward-model-plus-RLHF loop?
