"""
import torch
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
from torchvision.utils import save_image

from config import OUTPUT_DIR, SYN_IXI_SLICES, DIFFUSION_IMAGE_SIZE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = OUTPUT_DIR / "diffusion" / "ixi_ddpm"

    model = UNet2DModel(
        sample_size=DIFFUSION_IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    model.load_state_dict(torch.load(model_dir / "final.pt", map_location=device))
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    out_dir = SYN_IXI_SLICES / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_images = 500
    batch_size = 8
    steps = 250

    saved = 0
    while saved < n_images:
        current_bs = min(batch_size, n_images - saved)
        images = torch.randn((current_bs, 1, DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE), device=device)

        scheduler.set_timesteps(steps)

        for t in tqdm(scheduler.timesteps, leave=False, desc=f"Sampling batch starting {saved}"):
            with torch.no_grad():
                noise_pred = model(images, t).sample
            images = scheduler.step(noise_pred, t, images).prev_sample

        images = (images.clamp(-1, 1) + 1) / 2

        for i in range(current_bs):
            save_image(images[i], out_dir / f"syn_ixi_{saved + i:05d}.png")

        saved += current_bs

    print(f"Saved {n_images} synthetic IXI images to {out_dir}")


if __name__ == "__main__":
    main()

    """

import torch
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
from torchvision.utils import save_image

from config import OUTPUT_DIR, SYN_IXI_SLICES, DIFFUSION_IMAGE_SIZE


def build_model(device):
    model = UNet2DModel(
        sample_size=DIFFUSION_IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = OUTPUT_DIR / "diffusion" / "ixi_ddpm"

    model = build_model(device)
    model.load_state_dict(torch.load(model_dir / "final.pt", map_location=device))
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    out_dir = SYN_IXI_SLICES / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_images = 200
    batch_size = 8
    steps = 100

    saved = 0
    while saved < n_images:
        current_bs = min(batch_size, n_images - saved)
        images = torch.randn(
            (current_bs, 1, DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE),
            device=device
        )

        scheduler.set_timesteps(steps)

        for t in tqdm(scheduler.timesteps, leave=False, desc=f"Sampling batch starting {saved}"):
            with torch.no_grad():
                noise_pred = model(images, t).sample
            images = scheduler.step(noise_pred, t, images).prev_sample

        images = (images.clamp(-1, 1) + 1) / 2

        for i in range(current_bs):
            save_image(images[i], out_dir / f"syn_ixi_{saved + i:05d}.png")

        saved += current_bs

    print(f"Saved {n_images} synthetic images to {out_dir}")


if __name__ == "__main__":
    main()