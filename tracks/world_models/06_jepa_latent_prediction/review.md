# Review checklist

- Can you trace the shapes from images to context and target patch latents?
- Are context and target masks disjoint with a constant count per batch row?
- Is the target encoder a separate deep copy that starts equal to the online encoder?
- Do target latents remain outside the autograd graph?
- Does the predictor know which target position it is predicting?
- Can you explain why the EMA update happens after the optimizer step?
- Can you contrast latent prediction with the pixel reconstruction in `wm.01_vae`?
- Can you identify which parts of full I-JEPA were intentionally left out?
