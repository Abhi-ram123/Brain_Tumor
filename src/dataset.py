"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TumorClassificationDataset(Dataset):
    def __init__(self, csv_path, split, image_size=224, train=False):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if train:
            self.tfms = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.tfms = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("L")
        image = self.tfms(image)
        label = int(row["label"])
        return image, label, row["image_path"]

        """

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TumorClassificationDataset(Dataset):
    def __init__(self, csv_path, split, image_size=224, train=False):
        df=pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if train:
            self.tfms=transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.tfms=transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("L")
        image = self.tfms(image)
        label = int(row["label"])
        return image, label, row["image_path"]