import pandas as pd
from sklearn.model_selection import train_test_split

from config import METADATA_DIR, SPLITS_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO


def main():
    df = pd.read_csv(METADATA_DIR / "tumor_metadata.csv")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    val_fraction = VAL_RATIO / (1 - TRAIN_RATIO)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_fraction),
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df.to_csv(SPLITS_DIR / "tumor_classification.csv", index=False)

    ixi = pd.read_csv(METADATA_DIR / "ixi_metadata.csv")
    ixi.to_csv(SPLITS_DIR / "ixi_train_for_diffusion.csv", index=False)

    print("\nTumor split counts:")
    print(final_df.groupby(["split", "class_name"]).size())
    print(f"\nSaved classification CSV to {SPLITS_DIR / 'tumor_classification.csv'}")


if __name__ == "__main__":
    main()