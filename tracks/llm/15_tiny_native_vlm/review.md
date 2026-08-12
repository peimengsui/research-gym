# Review questions

- How can textual loss train the image patch projection?
- Why can text queries see all visual tokens but visual queries cannot see text?
- Why is the full `attention_mask` unsqueezed before use by multiple heads?
- Why are vocabulary logits produced only at text positions?
- Which token does the logit at text position `t` predict?
- Why does target validity use the shifted text mask?
- What is the difference between padding in attention and `ignore_index` in loss?
- What additions are needed to generate text one token at a time from an image?
