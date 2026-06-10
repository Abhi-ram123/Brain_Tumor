import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

OUTPUT_DIR="outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_history(history_file):
    with open(history_file, "r") as f:
        history=json.load(f)

    epochs=range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
    plt.close()

    # Accuracy curve
    if "val_acc" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(f"{OUTPUT_DIR}/accuracy_curve.png")
        plt.close()

    # F1 curve
    if "val_f1" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["val_f1"], label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Validation Macro F1")
        plt.legend()
        plt.grid()
        plt.savefig(f"{OUTPUT_DIR}/f1_curve.png")
        plt.close()


def plot_confusion_matrix(predictions_file):
    df = pd.read_csv(predictions_file)

    y_true=df["true_label"]
    y_pred=df["pred_label"]

    labels = sorted(list(set(y_true)))

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(ax=ax, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close()


def plot_class_distribution(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(8, 5))
    df["label"].value_counts().plot(kind="bar")

    plt.title("Class Distribution")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
    plt.close()


def plot_experiment_comparison():
    experiments={
        "Raw":0.96,
        "Denoised":0.23,
        "Mixed":0.90
    }

    names=list(experiments.keys())
    values=list(experiments.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)

    plt.title("Experiment Comparison (Macro F1)")
    plt.ylabel("Macro F1")

    plt.savefig(f"{OUTPUT_DIR}/experiment_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history",
        type=str,
        default=None
    )

    parser.add_argument(
        "--predictions",
        type=str,
        default=None
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.history:
        plot_training_history(args.history)

    if args.predictions:
        plot_confusion_matrix(args.predictions)

    if args.csv:
        plot_class_distribution(args.csv)

    plot_experiment_comparison()

    print(f"\nVisualizations saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()