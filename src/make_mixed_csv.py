import pandas as pd
from config import SPLITS_DIR


def main():
    raw_df = pd.read_csv(SPLITS_DIR / "tumor_classification.csv")
    den_df = pd.read_csv(SPLITS_DIR / "tumor_classification_denoised.csv")

    # Keep original validation and test from raw dataset
    raw_val_test = raw_df[raw_df["split"].isin(["val", "test"])].copy()

    # Use BOTH raw and denoised images for training
    raw_train = raw_df[raw_df["split"] == "train"].copy()
    den_train = den_df[den_df["split"] == "train"].copy()

    mixed_df = pd.concat([raw_train, den_train, raw_val_test], ignore_index=True)
    mixed_df.to_csv(SPLITS_DIR / "tumor_classification_mixed.csv", index=False)

    print("Saved mixed CSV to:", SPLITS_DIR / "tumor_classification_mixed.csv")
    print(mixed_df.groupby(["split", "class_name"]).size())


if __name__ == "__main__":
    main()