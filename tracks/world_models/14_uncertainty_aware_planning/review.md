# Review checklist

- Is softmax applied across mixture components rather than candidates or members?
- Are member-specific expected futures preserved before ensemble averaging?
- Does aleatoric uncertainty compare modes with their own member mean?
- Does epistemic uncertainty compare member means with the ensemble mean?
- Are both uncertainty values reduced to one scalar per candidate?
- Why does the familiar plan have high aleatoric but zero epistemic uncertainty?
- Why does the shortcut have low aleatoric but high epistemic uncertainty?
- Does the nominal score prefer the lower-effort shortcut?
- Does increasing epistemic risk aversion change the selected plan?
- Why is aleatoric uncertainty reported but not directly penalized here?
- When would a real planner still need to penalize a particular aleatoric mode?
- How does `wm.14` extend the multimodal predictions introduced in `wm.13`?
