# Concept: who may attend to whom

## From 1D validity to 2D attention

Lesson 13 produced:

```text
validity: (batch, total_tokens)
True  = real token
False = padding
```

Self-attention needs a different object:

```text
attention matrix: (batch, query, key)
True  = query may attend to key
False = query must ignore key
```

This lesson builds that matrix for a visual-prefix multimodal layout.

## Sequence layout

```text
[V0, V1, ..., Vk-1, S, T0, T1, ..., Tn-1]
|<------ prefix ----->|<---- text ---->|
```

The prefix length is `visual_token_count + 1` because the separator belongs with
the image side of the boundary.

## Three rules

1. **Bidirectional visual prefix.** Every prefix query may attend to every
   prefix key. Image patches should share information freely, like visual
   self-attention in lesson 12.
2. **Causal text.** Text query `Ti` may attend to the full prefix and to text
   keys `T0..Ti`, but not to future text.
3. **Padding is invisible.** No query may attend to a padded key. Padding query
   rows are all-False so they contribute no attention mass.

Together:

```text
allow[q, k] =
  validity[q]
  AND validity[k]
  AND (
        (q in prefix AND k in prefix)
        OR
        (q in text AND k <= q)
      )
```

## Why not plain causal over the whole sequence?

A fully lower-triangular mask would force early visual patches to ignore later
visual patches. That fights the bidirectional visual encoder idea. Treating the
image+separator region as an open prefix keeps vision dense while preserving
autoregressive text generation.

## Softmax detail

Disallowed scores are set to `-inf` before softmax. An all-False padding query
row would otherwise become NaN; zero those rows after softmax so padded slots
stay numerically inert.
