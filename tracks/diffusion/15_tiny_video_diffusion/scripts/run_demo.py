"""Train on one tiny moving-dot pattern and sample a four-frame clip."""

import torch

from implementation import motion_statistics, video_ddim_sample_loop
from provided import (
    SpatiotemporalDenoiser,
    make_diffusion_schedule,
    make_tiny_moving_dot_videos,
    video_noise_prediction_loss,
)


def render_frame(frame: torch.Tensor) -> str:
    pixels = frame.squeeze(0)
    return "\n".join(
        "".join("##" if value > 0.0 else ".." for value in row) for row in pixels
    )


def main() -> None:
    torch.manual_seed(15)
    torch.set_num_threads(1)
    target = make_tiny_moving_dot_videos(batch_size=1)
    training_videos = target.expand(8, -1, -1, -1, -1).clone()
    schedule = make_diffusion_schedule(24, beta_end=0.18)
    model = SpatiotemporalDenoiser(
        in_channels=1,
        hidden_channels=16,
        time_embedding_dim=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(150)

    model.train()
    for step in range(300):
        timesteps = torch.randint(
            0,
            schedule.num_timesteps,
            (training_videos.shape[0],),
            generator=generator,
        )
        loss = video_noise_prediction_loss(
            model,
            schedule,
            training_videos,
            timesteps,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in {0, 99, 199, 299}:
            print(f"training step {step + 1:>3}: loss={loss.item():.4f}")

    model.eval()
    initial_noise = torch.randn(
        target.shape,
        generator=torch.Generator().manual_seed(151),
    )
    sample = video_ddim_sample_loop(
        model,
        schedule,
        shape=tuple(target.shape),
        num_sampling_steps=12,
        initial_noise=initial_noise,
    )
    statistics = motion_statistics(sample.videos)

    print("\ngenerated clip:")
    for frame_index in range(sample.videos.shape[2]):
        print(f"\nframe {frame_index}")
        print(render_frame(sample.videos[0, :, frame_index]))
    print("\nsoft foreground centers:")
    print(statistics.centers[0].round(decimals=2))
    print(f"mean speed:       {statistics.mean_speed[0].item():.3f} pixels/frame")
    print(f"continuity error: {statistics.continuity_error[0].item():.3f}")
    print(f"reverse states:   {sample.trajectory.shape[0]}")


if __name__ == "__main__":
    main()
