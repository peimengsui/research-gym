# Review checklist

- Are forget-gate logits converted with `logsigmoid / normalizer`?
- Are all decay factors between zero and one?
- Is the gate data-dependent and one value per key channel?
- Does the gate decay old memory before the current write is added?
- Is the key/value update an outer product?
- Does the current query read from the updated memory?
- Do explicit parallel and recurrent forms agree?
- Does either form correctly continue from incoming matrix state?
- Is incoming state left unmodified?
- Does state size stay fixed as sequence length grows?
- Is there no softmax or key-sum denominator in the GLA recurrence?
- Is raw output normalized independently for each head?
- Is the SiLU output gate distinct from the memory forget gate?
- Do gradients reach both low-rank gate projections and all attention paths?
- Can you explain why the explicit oracle is not the optimized chunkwise kernel?
