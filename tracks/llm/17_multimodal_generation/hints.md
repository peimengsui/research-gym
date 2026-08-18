# Hints

## Hint 1

For total positions `query` and `key`, the mask structure is the union of:
`(query < prefix) & (key < prefix)` and
`(query >= prefix) & (key <= query)`.

## Hint 2

After `split_heads`, the token dimension is 2. Concatenate cached and new keys
with `torch.cat((cached_key, new_key), dim=2)`.

## Hint 3

The prefill sequence length is
`num_visual_tokens + 1 + prompt_token_ids.shape[1]`. Each layer cache must have
that same token length.

## Hint 4

Use `past_length = caches[0][0].shape[2]`. Create the position tensor on the
same device as `token_ids` and an all-true mask of shape
`(batch, 1, past_length + 1)`.

## Hint 5

The logits returned by prefill predict the first new token. Call `decode_step`
only after appending that token and only if another prediction is required.

## Hint 6

For EOS batching, use `torch.where(finished[:, None], eos_fill, next_token)` and
then update `finished` from the selected tokens.
