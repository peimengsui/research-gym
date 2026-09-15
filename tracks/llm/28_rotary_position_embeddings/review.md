# Review checklist

- Does every adjacent feature pair share exactly one inverse frequency?
- Do frequency indices and absolute positions use the requested device and dtype?
- Is the position-zero rotation the identity?
- Are cosine and sine broadcast over batch and head dimensions?
- Does rotation restore adjacent even/odd feature ordering?
- Are query and key vector norms preserved?
- Why do shared absolute-position shifts preserve query-key dot products?
- Does cached decoding use past key length as the new position offset?
- Are keys rotated before caching and left unchanged afterward?
- Are values cached without rotary transformation?
- Does a multi-token cached continuation use the correct rectangular causal mask?
- Do full-context and incremental attention outputs match?
