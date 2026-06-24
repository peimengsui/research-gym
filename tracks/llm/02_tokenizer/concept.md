# Concept: turning text into tokens

Neural networks operate on numbers, not strings. A tokenizer converts text
into token IDs and later converts those IDs back into readable text.

A character tokenizer is simple and lossless for known characters, but long
words require many tokens. Word tokenizers use fewer tokens, but struggle with
unseen words and enormous vocabularies.

Byte-pair encoding (BPE) starts small and learns useful larger pieces:

```text
b a n a n a
↓ merge frequent pair "a n"
b an an a
↓ merge frequent pair "b an"
ban an a
```

This lesson uses characters rather than raw bytes to keep the implementation
visible. The training loop still captures the central BPE idea:

1. initialize one token per known character
2. count adjacent token pairs in the corpus
3. merge the most frequent pair
4. repeat

The tokenizer records merges in training order. Encoding starts from character
IDs and replays those merges. Decoding concatenates the string represented by
each token.

## Unknown characters

The training vocabulary cannot contain every possible character. ID `0` is
reserved for `<unk>`. Any unseen character encodes to that ID, making failure
explicit instead of crashing.

Production tokenizers often begin from bytes so arbitrary Unicode text remains
reversible. Character-level unknown handling is a smaller stepping stone toward
that design.
