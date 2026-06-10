import pandas as pd
from tqdm import tqdm

from config import (
    IXI_RAW,
    IXI_SLICES,
    METADATA_DIR,
    IMAGE_SIZE,
    MIN_NONZERO_RATIO,
    RANDOM_SEED,
)
from utils import (
    seed_everything,
    load_nii,
    normalize_slice,
    resize_image,
    save_png,
    nonzero_ratio,
    extract_axial_slices,
)


def main():
    seed_everything(RANDOM_SEED)
    rows = []

    nii_files = sorted(list(IXI_RAW.rglob("*.nii")) + list(IXI_RAW.rglob("*.nii.gz")))
    print(f"Found {len(nii_files)} IXI files")

    for nii_path in tqdm(nii_files, desc="Processing IXI"):
        patient_id = nii_path.stem.replace(".nii", "")

        try:
            vol = load_nii(nii_path)
        except Exception as e:
            print(f"Skipping {nii_path} due to error: {e}")
            continue

        for idx, sl in extract_axial_slices(vol):
            sl = normalize_slice(sl)

            if nonzero_ratio(sl) < MIN_NONZERO_RATIO:
                continue

            sl = resize_image(sl, IMAGE_SIZE)
            out_path = IXI_SLICES / patient_id / f"slice_{idx:03d}.png"
            save_png(sl, out_path)

            rows.append(
                {
                    "dataset": "IXI",
                    "patient_id": patient_id,
                    "slice_idx": idx,
                    "image_path": str(out_path),
                    "label": -1,
                    "class_name": "healthy_ixi",
                    "source_file": str(nii_path),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = METADATA_DIR / "ixi_metadata.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()