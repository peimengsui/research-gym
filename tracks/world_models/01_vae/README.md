# Variational Autoencoder

Build a small fully connected variational autoencoder that compresses binary
vectors into a probabilistic latent representation and reconstructs them.

You will implement:

- an encoder network
- latent mean and log-variance heads
- the reparameterization trick
- a decoder network
- reconstruction and KL losses

Start the lesson from the repository root:

```bash
uv run rgym start wm.01_vae
cd workspace/wm.01_vae
uv run rgym test
```

Read `concept.md` for the idea and `guide.md` for the implementation sequence.
Edit only `implementation.py` in your workspace.

After completing the exercise, compare your code with the source reference at
`tracks/world_models/01_vae/solution.py`.
