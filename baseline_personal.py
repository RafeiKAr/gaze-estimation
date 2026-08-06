
# 0: Lib
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from tqdm import tqdm



# 1: calss PersonalGazeDataset
class PersonalGazeDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        dataset_size=None,
        read_all4once=True
    ):

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.read_all4once = read_all4once

        self.df = pd.read_csv(
            self.root_dir / "norm_labels.csv"
        )

        self.dataset_size = (
            dataset_size
            if dataset_size is not None
            else len(self.df)
        )


        if self.read_all4once:

            # Form des transformierten Bildes bestimmen
            img = Image.new("RGB", (500, 300))
            out = transform(img)

            self.images = torch.zeros(
                [self.dataset_size] + list(out.shape)
            )

            self.targets = torch.zeros(
                self.dataset_size,
                2
            )


            for idx in tqdm(range(self.dataset_size)):

                row = self.df.iloc[idx]

                image = Image.open(
                    self.root_dir
                    / "images"
                    / row["frame"]
                ).convert("RGB")


                self.targets[idx] = torch.tensor(
                    [
                        row["x"],
                        row["y"]
                    ],
                    dtype=torch.float32
                )


                if self.transform:
                    self.images[idx] = self.transform(image)
                else:
                    self.images[idx] = image

    def __len__(self):

        return self.dataset_size


    def __getitem__(self, idx):

        if self.read_all4once:

            return (
                self.images[idx],
                self.targets[idx]
            )

        row = self.df.iloc[idx]

        image = Image.open(
            self.root_dir
            / "images"
            / row["frame"]
        ).convert("RGB")

        target = torch.tensor(
            [
                row["x"],
                row["y"]
            ],
            dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return image, target



# 2: func diagonal_error
def diagonal_errors(model, loader, device):

    # Evaluation-Modus
    model.eval()

    targets_list = []
    preds_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds = model(images)
            targets_list.append(targets.cpu().numpy())
            preds_list.append(preds.cpu().numpy())

    targets_all = np.concatenate(targets_list, axis=0)
    predictions = np.concatenate(preds_list, axis=0)

    mae = mean_absolute_error(targets_all, predictions)

    rmse = np.sqrt(mean_squared_error(targets_all, predictions))

    diagonal_error_pct = np.round(100 * (rmse / np.sqrt(2)), 5)

    print(f"MAE : {mae:.4f} \t RMSE: {rmse:.4f} ")
    # print(f"RMSE: {rmse:.4f}")
    print(f"Diagonal-Error %: {diagonal_error_pct:.4f} %")

    return mae, rmse, diagonal_error_pct


def main():

    load_checkpoint = "./models/ResNet_optim-model_norm_subject_1000-200.path"
    person_root_dir = "./personalization/01"

    batch_size = 64

    # 3: Transformation
    # Transformationen:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2
        ),

        transforms.RandomGrayscale(p=0.05),

        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.5)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # 5: personal Dataset
    dataset = PersonalGazeDataset(
        root_dir=person_root_dir,
        transform=transform
    )

    # print(f"\nPersonalization_Dataset_Pfade: {dataset.root_dir}\n")


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True
    )


    # 6: load Model
    model = resnet18(weights=None)

    model_name = model.__class__.__name__
    # print(f"\nModel: {model_name}")

    # letzte Schicht ändern
    model.fc = nn.Sequential(
            nn.Linear(
                model.fc.in_features,
                512
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                128,
                2
            )
        )


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Gewicht laden
    checkpoint = torch.load(
        load_checkpoint,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)


    # 7: Output (Baseline_Personalization_Error)
    print(f"\nBaseline-Erorr:")
    baseline_mae, baseline_rmse, baseline_diag_pct = diagonal_errors(model, loader, device)
    print(f"baseline_diag_error={baseline_diag_pct:.4f}%")

    print(f"\nModel: {model_name}\n"
          f"personalization_pfad: '{person_root_dir}'\n"
          f"loading_saved_model(checkpoint): '{load_checkpoint}'\n"
          f"batch_size: {batch_size}\n")



if __name__ == "__main__":
    main()
