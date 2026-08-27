# Hints

1. Build one structural `(total, total)` mask, then combine it with per-row
   sequence validity.
2. Keep padded audio slots in the rectangular sequence; mask them rather than
   compacting each row.
3. Unified position IDs span audio slots, separator, and text.
4. Text logits start at `sequence.prefix_length`.
5. The complete generation/evaluation code below the TODOs is worth reading—it
   should look almost identical to the video-language version.
