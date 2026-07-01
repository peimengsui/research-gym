# Review questions

- Why do prompt tokens stay in `input_ids` even when their labels are ignored?
- Why is the label mask shifted with `train_mask[1:]`?
- Should the `<assistant>` role token itself receive loss? Why or why not?
- What changes when a real tokenizer splits one word into multiple subword
  tokens?
