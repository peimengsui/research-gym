"""Overfit one tiny video-noise batch and inspect temporal coupling."""

import torch

from implementation import SpatiotemporalDenoiser, video_noise_prediction_loss
from provided import make_diffusion_schedule, make_tiny_videos


def main() -> None:
    torch.manual_seed(14)
    torch.set_num_threads(1)
    videos = make_tiny_videos(batch_size=4, num_frames=4, image_size=8)
    schedule = make_diffusion_schedule(12)
    timesteps = torch.tensor([3, 5, 7, 9])
    fixed_noise = torch.randn_like(videos)
    model = SpatiotemporalDenoiser(
        in_channels=1,
        hidden_channels=8,
        time_embedding_dim=16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)

    with torch.no_grad():
        initial_loss = video_noise_prediction_loss(
            model,
            schedule,
            videos,
            timesteps,
            fixed_noise,
        )

    model.train()
    for _ in range(80):
        loss = video_noise_prediction_loss(
            model,
            schedule,
            videos,
            timesteps,
            fixed_noise,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = video_noise_prediction_loss(
            model,
            schedule,
            videos,
            timesteps,
            fixed_noise,
        )
        baseline = torch.zeros(1, 1, 4, 8, 8)
        perturbed = baseline.clone()
        perturbed[:, :, 1, 3:5, 3:5] = 1.0
        baseline_prediction = model(baseline, torch.tensor([6]))
        perturbed_prediction = model(perturbed, torch.tensor([6]))
        neighbor_change = (
            (perturbed_prediction[:, :, 2] - baseline_prediction[:, :, 2]).abs().mean()
        )

    print(f"video shape:       {tuple(videos.shape)}")
    print(f"prediction shape:  {tuple(baseline_prediction.shape)}")
    print(f"initial fixed-noise loss: {initial_loss.item():.4f}")
    print(f"final fixed-noise loss:   {final_loss.item():.4f}")
    print(f"next-frame response to frame-1 perturbation: {neighbor_change.item():.6f}")
    print("A nonzero response confirms that the denoiser couples neighboring frames.")


if __name__ == "__main__":
    main()
