"""
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from config import SPLITS_DIR, OUTPUT_DIR, DIFFUSION_IMAGE_SIZE, RANDOM_SEED
from utils import seed_everything


class IXIDiffusionDataset(Dataset):
    def __init__(self, csv_path, image_size):
        self.df = pd.read_csv(csv_path)
        self.tfms = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]["image_path"]).convert("L")
        return self.tfms(img)


def main():
    seed_everything(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = OUTPUT_DIR / "diffusion" / "ixi_ddpm"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = IXIDiffusionDataset(SPLITS_DIR / "ixi_train_for_diffusion.csv", DIFFUSION_IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    model = UNet2DModel(
        sample_size=DIFFUSION_IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 20
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        for batch in pbar:
            clean = batch.to(device)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean.shape[0],),
                device=device,
            ).long()

            noisy = noise_scheduler.add_noise(clean, noise, timesteps)
            pred = model(noisy, timesteps).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

        torch.save(model.state_dict(), save_dir / f"epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1}: avg_loss = {running_loss / len(loader):.6f}")

    torch.save(model.state_dict(), save_dir / "final.pt")
    print(f"Saved final model to {save_dir}")


if __name__ == "__main__":
    main()
"""

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from config import SPLITS_DIR, OUTPUT_DIR, DIFFUSION_IMAGE_SIZE, RANDOM_SEED
from utils import seed_everything


class IXIDiffusionDataset(Dataset):
    def __init__(self, csv_path, image_size):
        self.df = pd.read_csv(csv_path)
        self.tfms = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]["image_path"]).convert("L")
        return self.tfms(img)


def main():
    seed_everything(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = OUTPUT_DIR / "diffusion" / "ixi_ddpm"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = IXIDiffusionDataset(
        SPLITS_DIR / "ixi_train_for_diffusion.csv",
        DIFFUSION_IMAGE_SIZE,
    )

    max_samples = 5000
    if len(dataset) > max_samples:
        dataset = Subset(dataset, range(max_samples))

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = UNet2DModel(
        sample_size=DIFFUSION_IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 3
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        for batch in pbar:
            clean = batch.to(device)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean.shape[0],),
                device=device,
            ).long()

            noisy = noise_scheduler.add_noise(clean, noise, timesteps)
            pred = model(noisy, timesteps).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), save_dir / f"epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1}: avg_loss = {running_loss / len(loader):.6f}")

    torch.save(model.state_dict(), save_dir / "final.pt")
    print(f"Saved diffusion model to {save_dir}")


if __name__ == "__main__":
    main()