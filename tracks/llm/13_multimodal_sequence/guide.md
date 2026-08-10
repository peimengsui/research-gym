# Guide

Open `implementation.py`. The finished vision stack lives in `provided.py`; you
do not need to modify it. Complete the five TODOs in order.

## 1. Mark text padding

Compare `text_token_ids` with `pad_token_id`. Real tokens should be `True` and
padding should be `False`. Preserve the original batch and text axes.

## 2. Build the combined validity mask

Create a boolean tensor of ones for `num_visual_tokens + 1` positions. The extra
position is the separator. Use the text mask's batch size and device, then
concatenate the text mask after this prefix.

This mask does not yet describe who may attend to whom. It only describes which
sequence positions contain valid content.

## 3. Assign token types

Create three long tensors on the text IDs' device:

```text
visual positions  → VISUAL_TOKEN_TYPE
separator         → SEPARATOR_TOKEN_TYPE
text positions    → TEXT_TOKEN_TYPE
```

Concatenate them in exactly the same order as the future embeddings.

## 4. Create multimodal parameters

The tests use these attribute names directly:

- `self.text_embedding`
- `self.separator_token`
- `self.token_type_embedding`
- `self.sequence_position_embedding`

The separator is an `nn.Parameter` with shape `(1, 1, embed_dim)`. Initialize it
with `nn.init.normal_` and a small standard deviation such as `0.02`.

The full position table needs
`num_visual_tokens + 1 + max_text_tokens` rows.

## 5. Assemble the sequence

Encode images and embed text IDs. Expand—not repeat—the separator over the
batch, then concatenate content in this exact order:

```text
[visual_tokens, separator, text_tokens]
```

Build token-type IDs and positions `0..total_tokens-1`. Add both embedding
tables to the content sequence. Finally, construct the combined validity mask
and return all fields in `MultimodalSequence`.

Do not remove padded text slots. Keeping the sequence rectangular is what makes
batching straightforward.

## Run

```bash
uv run rgym test
uv run rgym run
```
