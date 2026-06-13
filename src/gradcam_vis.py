"""
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import TumorClassificationDataset
from config import SPLITS_DIR, IMAGE_SIZE, OUTPUT_DIR, CLASS_NAMES


def build_model(num_classes=4):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def denorm(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy().transpose(1, 2, 0)
    x = (x * 0.5) + 0.5
    return np.clip(x, 0, 1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    csv_name = "tumor_classification_denoised.csv" if args.use_denoised else "tumor_classification.csv"
    ds = TumorClassificationDataset(SPLITS_DIR / csv_name, split="test", image_size=IMAGE_SIZE, train=False)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    out_dir = OUTPUT_DIR / "figures" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(12, len(ds))):
        x, y, _ = ds[i]
        input_tensor = x.unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        rgb_img = denorm(x)

        vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        out_file = out_dir / f"cam_{i:02d}_label_{CLASS_NAMES[y]}.png"
        cv2.imwrite(str(out_file), vis)

    print(f"Saved Grad-CAM images to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_denoised", action="store_true")
    main(parser.parse_args())

    """

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import TumorClassificationDataset
from config import SPLITS_DIR, IMAGE_SIZE, OUTPUT_DIR, CLASS_NAMES


def build_model(num_classes=4):
    model= models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def denorm(x: torch.Tensor) -> np.ndarray:
    x=x.detach().cpu().numpy().transpose(1, 2, 0)
    x = (x * 0.5) + 0.5
    return np.clip(x, 0, 1)


def main(args):
    device="cuda" if torch.cuda.is_available() else "cpu"

    model=build_model(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    csv_name="tumor_classification_denoised.csv" if args.use_denoised else "tumor_classification.csv"
    ds = TumorClassificationDataset(SPLITS_DIR / csv_name, split="test", image_size=IMAGE_SIZE, train=False)

    target_layers=[model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    out_dir = OUTPUT_DIR / "figures" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(12, len(ds))):
        x, y, _ = ds[i]
        input_tensor = x.unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        rgb_img = denorm(x)
        vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_dir / f"cam_{i:02d}_label_{CLASS_NAMES[y]}.png"), vis)

    print(f"Saved Grad-CAM images to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_denoised", action="store_true")
    main(parser.parse_args())