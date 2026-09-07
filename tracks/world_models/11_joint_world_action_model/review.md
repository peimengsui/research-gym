# Review checklist

- Which state tensor is shared by the action and dynamics branches?
- Do both objectives send gradients into `state_encoder`?
- Why does language condition action prediction but not physical dynamics?
- Are transition actions normalized before entering the dynamics model?
- Why is latent dynamics implemented as a residual update?
- Does the action head return `[batch, chunk_size, action_dim]`?
- Is the next-state target detached from gradient computation?
- Are action and latent component losses reported before weighting?
- How does this differ from the reactive `wm.10` policy?
- What interface can `wm.12` reuse to imagine candidate action sequences?
