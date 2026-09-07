# Vision, Language, and Action Chunks

Build a tiny reactive policy that maps a short visual history and a language
instruction directly to a chunk of continuous actions.

## What you will build

- per-dimension action normalization and denormalization
- fusion of provided video and language features
- a bounded continuous action-chunk head
- masked behavior-cloning loss for variable-length chunks

The visual, video, and language encoders, input validation, synthetic expert
demonstrations, and optimizer step are provided. The exercise stays focused on
the new VLA pieces instead of repeating earlier multimodal lessons.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```

This is a small reactive imitation-learning baseline inspired by systems such
as [RT-1](https://arxiv.org/abs/2212.06817) and
[OpenVLA](https://arxiv.org/abs/2406.09246). It does not predict future visual
states or plan with a world model; that distinction prepares the comparison in
`wm.11_joint_world_action_model`.
