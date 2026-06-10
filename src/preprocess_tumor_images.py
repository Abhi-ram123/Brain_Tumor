import cv2
import pandas as pd
from tqdm import tqdm

from config import (
    TUMOR_RAW,
    TUMOR_PROCESSED,
    METADATA_DIR,
    IMAGE_SIZE,
    CLASS_NAMES,
    CLASS_TO_IDX,
    RANDOM_SEED,
)
from utils import seed_everything, normalize_slice, resize_image, save_png

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    seed_everything(RANDOM_SEED)
    rows = []

    for class_name in CLASS_NAMES:
        class_dir = TUMOR_RAW / class_name

        if not class_dir.exists():
            print(f"Missing class folder: {class_dir}")
            continue

        image_files = [p for p in class_dir.rglob("*") if p.suffix.lower() in VALID_EXTS]
        print(f"{class_name}: {len(image_files)} files")

        for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            img = normalize_slice(img)
            img = resize_image(img, IMAGE_SIZE)

            out_path = TUMOR_PROCESSED / class_name / (img_path.stem + ".png")
            save_png(img, out_path)

            rows.append(
                {
                    "dataset": "tumor_images",
                    "patient_id": img_path.stem,
                    "slice_idx": 0,
                    "image_path": str(out_path),
                    "label": CLASS_TO_IDX[class_name],
                    "class_name": class_name,
                    "source_file": str(img_path),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = METADATA_DIR / "tumor_metadata.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} tumor image records to {out_csv}")


if __name__ == "__main__":
    main()