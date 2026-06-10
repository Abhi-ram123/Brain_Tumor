import random
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    return np.asarray(data, dtype=np.float32)


def normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max()>0:
        x = x / x.max()
    return x


def resize_image(x: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)


def save_png(x: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), img)


def nonzero_ratio(x: np.ndarray) -> float:
    return float(np.count_nonzero(x) / x.size)


def extract_axial_slices(volume: np.ndarray):
    for i in range(volume.shape[2]):
        yield i, volume[:, :, i]