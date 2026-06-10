"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from config import SPLITS_DIR, IMAGE_SIZE, CLASS_NAMES
from dataset import TumorClassificationDataset


def build_model(num_classes=4):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_name = "tumor_classification_denoised.csv" if args.use_denoised else "tumor_classification.csv"
    csv_path = SPLITS_DIR / csv_name

    test_ds = TumorClassificationDataset(csv_path, split="test", image_size=IMAGE_SIZE, train=False)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()

            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_denoised", action="store_true")
    main(parser.parse_args())

    """
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from config import SPLITS_DIR, IMAGE_SIZE, CLASS_NAMES
from dataset import TumorClassificationDataset


def build_model(num_classes=4):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_name = "tumor_classification_denoised.csv" if args.use_denoised else "tumor_classification.csv"
    csv_path = SPLITS_DIR / csv_name

    test_ds = TumorClassificationDataset(csv_path, split="test", image_size=IMAGE_SIZE, train=False)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()

            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_denoised", action="store_true")
    main(parser.parse_args())

    """

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from config import SPLITS_DIR, IMAGE_SIZE, CLASS_NAMES
from dataset import TumorClassificationDataset


def build_model(num_classes=4):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.csv_name is not None:
        csv_path = SPLITS_DIR / args.csv_name
    else:
        csv_name = "tumor_classification_denoised.csv" if args.use_denoised else "tumor_classification.csv"
        csv_path = SPLITS_DIR / csv_name

    test_ds = TumorClassificationDataset(csv_path, split="test", image_size=IMAGE_SIZE, train=False)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()

            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_denoised", action="store_true")
    parser.add_argument("--csv_name", type=str, default=None)
    main(parser.parse_args())