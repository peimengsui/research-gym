# Review questions

- Why must image height and width be divisible by the patch size here?
- Which permutation makes patches follow row-major sequence order?
- Why can `unpatchify(patchify(image))` reconstruct the image exactly?
- Which parameters are shared across every patch and which depend on position?
- Why are position embeddings needed even though patches contain pixel values?
- How does choosing a smaller patch size change sequence length and attention cost?
- What additional work would variable-resolution image batches require?
