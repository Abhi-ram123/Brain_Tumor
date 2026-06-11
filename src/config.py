from pathlib import Path

# Project root
ROOT=Path(__file__).resolve().parents[1]

# Output directories (these will be created automatically)
DATA_DIR=ROOT / "data"
PROCESSED_DIR=DATA_DIR / "processed"
METADATA_DIR=PROCESSED_DIR / "metadata"
SPLITS_DIR=DATA_DIR / "splits"
OUTPUT_DIR=ROOT / "outputs"

# ===== IMPORTANT: USE YOUR LOCAL DATASET PATHS =====

#IXI_RAW = Path(r"C:\MTECH\ROP\NII dataset\IXI")
#TUMOR_RAW = Path(r"C:\MTECH\ROP\Tumor Dataset")

# Raw dataset paths (UPDATED)

IXI_RAW=Path(r"C:\MTECH\ROP\NII dataset\NII_DATASET_UPDATED\NII_DATASET_UPDATED_1")

TUMOR_RAW=Path(r"C:\MTECH\ROP\multi-modal-mri-and-mask-synthesis-with-conditional-slice-based-ldm-main\Training")

# =================================================

# Processed data folders
IXI_SLICES=PROCESSED_DIR / "ixi_slices"
TUMOR_PROCESSED=PROCESSED_DIR / "tumor_images_processed"
SYN_IXI_SLICES=PROCESSED_DIR / "synthetic_ixi"
DENOISED_TUMOR=PROCESSED_DIR / "tumor_images_denoised"

# Settings
IMAGE_SIZE=224
DIFFUSION_IMAGE_SIZE = 128
RANDOM_SEED = 42
MIN_NONZERO_RATIO = 0.10

# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Classes
CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Create required folders automatically
for p in [
    PROCESSED_DIR,
    METADATA_DIR,
    SPLITS_DIR,
    OUTPUT_DIR,
    IXI_SLICES,
    TUMOR_PROCESSED,
    SYN_IXI_SLICES,
    DENOISED_TUMOR,
]:
    p.mkdir(parents=True, exist_ok=True)