# Hints

1. After temporal `unfold`, the new within-tubelet dimension is appended at the
   end rather than inserted beside the frame grid dimension.
2. A useful extraction permutation is the one that produces grid dimensions
   first and content dimensions second.
3. `repeat_interleave(spatial_token_count)` creates temporal IDs in the required
   token order.
4. `spatial_ids.repeat(temporal_token_count)` creates the matching spatial IDs.
