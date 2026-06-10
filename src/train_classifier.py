"""
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from config import SPLITS_DIR, OUTPUT_DIR, IMAGE_SIZE, RANDOM_SEED, CLASS_NAMES
from utils import seed_everything
from dataset import TumorClassificationDataset


def build_model(num_classes=4):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def main(args):
    seed_everything(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_name = "tumor_classification_denoised.csv" if args.use_denoised else "tumor_classification.csv"
    csv_path = SPLITS_DIR / csv_name

    train_ds = TumorClassificationDataset(csv_path, split="train", image_size=IMAGE_SIZE, train=True)
    val_ds = TumorClassificationDataset(csv_path, split="val", image_size=IMAGE_SIZE, train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    save_dir = OUTPUT_DIR / "classifier" / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    epochs = 15
    history = []

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()))

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch + 1
        history.append(metrics)

        print(f"Validation metrics: {metrics}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), save_dir / "best.pt")

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--use_denoised", action="store_true")
    main(parser.parse_args())

    """
"""
import argparse
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from config import SPLITS_DIR, OUTPUT_DIR, IMAGE_SIZE, RANDOM_SEED, CLASS_NAMES
from utils import seed_everything
from dataset import TumorClassificationDataset


def build_model(num_classes=4):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def get_class_weights(csv_path, device):
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]

    class_counts = train_df["label"].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def main(args):
    seed_everything(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_name = "tumor_classification_denoised.csv" if args.use_denoised else "tumor_classification.csv"
    csv_path = SPLITS_DIR / csv_name

    train_ds = TumorClassificationDataset(csv_path, split="train", image_size=IMAGE_SIZE, train=True)
    val_ds = TumorClassificationDataset(csv_path, split="val", image_size=IMAGE_SIZE, train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    class_weights = get_class_weights(csv_path, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    save_dir = OUTPUT_DIR / "classifier" / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    epochs = 25
    history = []

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        running_loss = 0.0

        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["f1"])

        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running_loss / len(train_loader)
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        history.append(metrics)

        print(f"Validation metrics: {metrics}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), save_dir / "best.pt")

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--use_denoised", action="store_true")
    main(parser.parse_args())
  """  

import argparse
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from config import SPLITS_DIR, OUTPUT_DIR, IMAGE_SIZE, RANDOM_SEED, CLASS_NAMES
from utils import seed_everything
from dataset import TumorClassificationDataset


def build_model(num_classes=4):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def get_class_weights(csv_path, device):
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]

    class_counts = train_df["label"].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def main(args):
    seed_everything(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.csv_name is not None:
        csv_path = SPLITS_DIR / args.csv_name
    else:
        csv_name = "tumor_classification_denoised.csv" if args.use_denoised else "tumor_classification.csv"
        csv_path = SPLITS_DIR / csv_name

    train_ds = TumorClassificationDataset(csv_path, split="train", image_size=IMAGE_SIZE, train=True)
    val_ds = TumorClassificationDataset(csv_path, split="val", image_size=IMAGE_SIZE, train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    class_weights = get_class_weights(csv_path, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    save_dir = OUTPUT_DIR / "classifier" / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    epochs = 25
    history = []

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        running_loss = 0.0

        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["f1"])

        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running_loss / len(train_loader)
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        history.append(metrics)

        print(f"Validation metrics: {metrics}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), save_dir / "best.pt")

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--use_denoised", action="store_true")
    parser.add_argument("--csv_name", type=str, default=None)
    main(parser.parse_args())