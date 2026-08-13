# Review questions

- Why are image and user-prompt inputs valid context but not supervised labels?
- Why should `<eos>` be part of the assistant target region?
- Why is the assistant mask shifted before it is applied to labels?
- Why does truncation keep one more full token than the model input width?
- How do `attention_mask` and `IGNORE_INDEX` serve different purposes?
- Why must images have a common shape in this simple collator?
- What response-preserving truncation policy might a production pipeline use?
- How can assistant-only text loss still train the native image encoder?
