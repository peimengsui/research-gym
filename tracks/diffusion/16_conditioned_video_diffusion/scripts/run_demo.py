"""Train tiny motion prompts and sample two futures from one first frame."""

import torch

from implementation import conditioned_video_ddim_sample_loop, direction_alignment_score
from provided import (
    DIRECTION_NAMES,
    ConditionedVideoDenoiser,
    conditioned_video_noise_prediction_loss,
    frame_centers,
    make_conditioned_moving_dot_batch,
    make_diffusion_schedule,
)


def render_frame(frame: torch.Tensor) -> str:
    pixels = frame.squeeze(0)
    return "\n".join(
        "".join("##" if value > 0.5 else ".." for value in row) for row in pixels
    )


def main() -> None:
    torch.manual_seed(16)
    torch.set_num_threads(1)
    training_batch = make_conditioned_moving_dot_batch(batch_size=4)
    schedule = make_diffusion_schedule(24, beta_end=0.3)
    model = ConditionedVideoDenoiser(
        hidden_channels=16,
        condition_dim=32,
        num_attention_heads=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(160)

    model.train()
    for step in range(700):
        loss = conditioned_video_noise_prediction_loss(
            model,
            schedule,
            training_batch.videos,
            training_batch.text_token_ids,
            training_batch.text_attention_mask,
            condition_drop_probability=0.1,
            foreground_weight=20.0,
            generator=generator,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in {0, 199, 399, 699}:
            print(f"training step {step + 1:>3}: loss={loss.item():.4f}")

    # Direction IDs 0 and 3 are right and up. Their clean first frames are
    # identical, and both samples receive exactly the same initial Gaussian noise.
    demo_indices = torch.tensor([0, 3])
    first_frames = training_batch.first_frames[demo_indices]
    prompts = training_batch.text_token_ids[demo_indices]
    prompt_mask = training_batch.text_attention_mask[demo_indices]
    directions = training_batch.direction_ids[demo_indices]
    shared_noise = (
        torch.randn(
            (1, 1, 4, 8, 8),
            generator=torch.Generator().manual_seed(161),
        )
        .expand(2, -1, -1, -1, -1)
        .clone()
    )
    shared_frame_noise = (
        torch.randn(
            (1, 1, 8, 8),
            generator=torch.Generator().manual_seed(162),
        )
        .expand(2, -1, -1, -1)
        .clone()
    )

    model.eval()
    sample = conditioned_video_ddim_sample_loop(
        model,
        schedule,
        first_frames,
        prompts,
        prompt_mask,
        num_frames=4,
        num_sampling_steps=12,
        guidance_scale=3.0,
        initial_noise=shared_noise,
        first_frame_noise=shared_frame_noise,
    )
    centers = frame_centers(sample.videos)
    alignment = direction_alignment_score(sample.videos, directions)

    for video_index, direction_id in enumerate(directions):
        direction = DIRECTION_NAMES[int(direction_id)]
        print(f"\ncondition: move {direction}")
        for frame_index in range(sample.videos.shape[2]):
            print(f"\nframe {frame_index}")
            print(render_frame(sample.videos[video_index, :, frame_index]))
        print(f"centers: {centers[video_index].round(decimals=2)}")
        print(f"direction alignment: {alignment[video_index].item():.3f}")

    first_frame_error = (sample.videos[:, :, 0] - first_frames).abs().max()
    print(f"\nmaximum preserved-first-frame error: {first_frame_error.item():.1f}")


if __name__ == "__main__":
    main()
