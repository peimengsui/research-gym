# Hints

1. A prefix/text structural mask can be expressed with query and key index grids.
2. Mask scores before softmax, not probabilities afterward.
3. `targets[:, :-1] = text_token_ids[:, 1:]` performs the next-token shift.
4. Text logits begin after `video_token_count + 1` prefix positions.
5. Candidate token zero is predicted by the final prompt position, so scoring
   starts at `prompt_length - 1`.
6. Save `was_training = model.training`, call `eval()`, and restore with
   `model.train(was_training)` after evaluation.
