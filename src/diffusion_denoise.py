"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from config import SPLITS_DIR, DENOISED_TUMOR, OUTPUT_DIR, DIFFUSION_IMAGE_SIZE


def build_model(device):
    model = UNet2DModel(
        sample_size=DIFFUSION_IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    model.load_state_dict(torch.load(OUTPUT_DIR / "diffusion" / "ixi_ddpm" / "final.pt", map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = build_model(device)

    df = pd.read_csv(SPLITS_DIR / "tumor_classification.csv")

    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    new_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Denoising tumor images"):
        img = Image.open(row["image_path"]).convert("L")
        x = tfm(img).unsqueeze(0).to(device)

        noise = torch.randn_like(x) * 0.1
        noisy = x + noise
        noisy = noisy.clamp(-1, 1)

        scheduler.set_timesteps(args.steps)
        sample = noisy

        for t in scheduler.timesteps:
            with torch.no_grad():
                pred_noise = model(sample, t).sample
            sample = scheduler.step(pred_noise, t, sample).prev_sample

        out = (sample.squeeze().detach().cpu().numpy() + 1) / 2
        out = np.clip(out, 0, 1)
        out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA)

        out_path = DENOISED_TUMOR / row["class_name"] / Path(row["image_path"]).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), (out * 255).astype(np.uint8))

        new_row = row.to_dict()
        new_row["image_path"] = str(out_path)
        new_rows.append(new_row)

    out_csv = SPLITS_DIR / "tumor_classification_denoised.csv"
    pd.DataFrame(new_rows).to_csv(out_csv, index=False)
    print(f"Saved denoised CSV to {out_csv}")


if __name__ == "__main__":
    main()

"""

"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from config import SPLITS_DIR, DENOISED_TUMOR, OUTPUT_DIR, DIFFUSION_IMAGE_SIZE


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

    model.load_state_dict(
        torch.load(
            OUTPUT_DIR / "diffusion" / "ixi_ddpm" / "final.pt",
            map_location=device
        )
    )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = build_model(device)

    df = pd.read_csv(SPLITS_DIR / "tumor_classification.csv")

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    new_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Denoising tumor images"):
        img = Image.open(row["image_path"]).convert("L")
        x = tfm(img).unsqueeze(0).to(device)

        noise = torch.randn_like(x) * 0.1
        sample = (x + noise).clamp(-1, 1)

        scheduler.set_timesteps(args.steps)

        for t in scheduler.timesteps:
            with torch.no_grad():
                pred_noise = model(sample, t).sample
            sample = scheduler.step(pred_noise, t, sample).prev_sample

        out = (sample.squeeze().detach().cpu().numpy() + 1) / 2
        out = np.clip(out, 0, 1)
        out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA)

        out_path = DENOISED_TUMOR / row["class_name"] / Path(row["image_path"]).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), (out * 255).astype(np.uint8))

        new_row = row.to_dict()
        new_row["image_path"] = str(out_path)
        new_rows.append(new_row)

    out_csv = SPLITS_DIR / "tumor_classification_denoised.csv"
    pd.DataFrame(new_rows).to_csv(out_csv, index=False)
    print(f"Saved denoised CSV to {out_csv}")


if __name__ == "__main__":
    main()

    """

"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from config import SPLITS_DIR, DENOISED_TUMOR, OUTPUT_DIR, DIFFUSION_IMAGE_SIZE


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

    model.load_state_dict(
        torch.load(
            OUTPUT_DIR / "diffusion" / "ixi_ddpm" / "final.pt",
            map_location=device
        )
    )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = build_model(device)

    df = pd.read_csv(SPLITS_DIR / "tumor_classification.csv")

    # denoise only training images
    df = df[df["split"] == "train"].reset_index(drop=True)

    # quick subset for faster testing
    max_images = 500
    if len(df) > max_images:
        df = df.iloc[:max_images].reset_index(drop=True)

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    new_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Denoising tumor images"):
        img = Image.open(row["image_path"]).convert("L")
        x = tfm(img).unsqueeze(0).to(device)

        noise = torch.randn_like(x) * 0.1
        sample = (x + noise).clamp(-1, 1)

        scheduler.set_timesteps(args.steps)

        for t in scheduler.timesteps:
            with torch.no_grad():
                pred_noise = model(sample, t).sample
            sample = scheduler.step(pred_noise, t, sample).prev_sample

        out = (sample.squeeze().detach().cpu().numpy() + 1) / 2
        out = np.clip(out, 0, 1)
        out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA)

        out_path = DENOISED_TUMOR / row["class_name"] / Path(row["image_path"]).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), (out * 255).astype(np.uint8))

        new_row = row.to_dict()
        new_row["image_path"] = str(out_path)
        new_rows.append(new_row)

    out_csv = SPLITS_DIR / "tumor_classification_denoised.csv"
    pd.DataFrame(new_rows).to_csv(out_csv, index=False)
    print(f"Saved denoised CSV to {out_csv}")


if __name__ == "__main__":
    main()

"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from config import SPLITS_DIR, DENOISED_TUMOR, OUTPUT_DIR, DIFFUSION_IMAGE_SIZE


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

    model.load_state_dict(
        torch.load(
            OUTPUT_DIR / "diffusion" / "ixi_ddpm" / "final.pt",
            map_location=device
        )
    )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=2870)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = build_model(device)

    full_df = pd.read_csv(SPLITS_DIR / "tumor_classification.csv")

    # Denoise only training images
    train_df = full_df[full_df["split"] == "train"].reset_index(drop=True)

    # Faster testing subset
    if len(train_df) > args.max_images:
        train_df = train_df.iloc[:args.max_images].reset_index(drop=True)

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    new_train_rows = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Denoising train images"):
        img = Image.open(row["image_path"]).convert("L")
        x = tfm(img).unsqueeze(0).to(device)

        noise = torch.randn_like(x) * 0.1
        sample = (x + noise).clamp(-1, 1)

        scheduler.set_timesteps(args.steps)

        for t in scheduler.timesteps:
            with torch.no_grad():
                pred_noise = model(sample, t).sample
            sample = scheduler.step(pred_noise, t, sample).prev_sample

        out = (sample.squeeze().detach().cpu().numpy() + 1) / 2
        out = np.clip(out, 0, 1)
        out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA)

        out_path = DENOISED_TUMOR / row["class_name"] / Path(row["image_path"]).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), (out * 255).astype(np.uint8))

        new_row = row.to_dict()
        new_row["image_path"] = str(out_path)
        new_train_rows.append(new_row)

    denoised_train_df = pd.DataFrame(new_train_rows)

    # Keep original validation and test rows unchanged
    val_test_df = full_df[full_df["split"].isin(["val", "test"])].copy()

    # Combine denoised train + original val/test
    final_df = pd.concat([denoised_train_df, val_test_df], ignore_index=True)

    out_csv = SPLITS_DIR / "tumor_classification_denoised.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Saved denoised CSV to {out_csv}")
    print(final_df.groupby("split").size())


if __name__ == "__main__":
    main()