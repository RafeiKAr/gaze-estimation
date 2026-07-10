# 1: Bib
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)
from datetime import datetime


# 2: Dataset Class: transfer CSV in PyTorch.
class GazeDataset(Dataset):

    def __init__(self, csv_file, transform=None, dataset_size=None, read_all4once=True):

        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.dataset_size = dataset_size if dataset_size is not None else len(self.df)
        self.read_all4once = read_all4once

        if self.read_all4once:
            img = Image.new("RGB", (500, 300))  # any size
            out = transform(img)
            self.images = torch.zeros([self.dataset_size] + list(out.shape))
            self.targets = torch.zeros(self.dataset_size, 2)

        for idx in tqdm(range(self.dataset_size)):

            row = self.df.iloc[idx]

            image = Image.open(
                row["image_name"]
            ).convert("RGB")

            self.targets[idx] = torch.tensor(
                [row["x"], row["y"]],
                dtype=torch.float32
            )

            if self.transform:
                self.images[idx] = self.transform(image)
            else:
                self.images[idx] = image

    def __len__(self):

        return self.dataset_size

    def __getitem__(self, idx):

        return self.images[idx], self.targets[idx]


# 3: func-diagonla-error:
def find_screen_size():
    """Suche die Bildschirmbreite aus einer Datei mit screen_w und screen_h"""
    candidates = [
        # "norm_labels.py",
        # "dataset/norm_labels.py",
        # "norm_labels.csv",
        "dataset/norm_labels.csv",
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue

        if path.endswith(".csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "screen_w" and "screen_h" in df.columns:
                return float(df["screen_w"].iloc[0]), float(df["screen_h"].iloc[0])
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            match = re.search(r"screen_w\s*=\s*([0-9.]+)", content)
            if match:
                return float(match.group(1))

    for root, _, files in os.walk("."):
        for name in files:
            if "norm_labels" not in name.lower():
                continue
            full_path = os.path.join(root, name)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            match = re.search(r"screen_w\s*=\s*([0-9.]+)", content)
            if match:
                return float(match.group(1))

    raise ValueError("screen_w konnte nicht gefunden werden. Bitte Datei mit 'screen_w = ...' angeben.")


def diagonal_errors(model, loader, device):
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

    screen_w, screen_h = find_screen_size()
    mae = mean_absolute_error(targets_all, predictions)
    rmse = np.sqrt(mean_squared_error(targets_all, predictions))
    diagonal_error_pct = 100 * np.round(rmse / np.sqrt(2 * screen_w * screen_h), 3)

    print(f"MAE : {mae:.4f} \t RMSE: {rmse:.4f} ")
    # print(f"RMSE: {rmse:.4f}")
    print(f"Diagonal-Error %: {diagonal_error_pct:.3f} %")

    return mae, rmse, diagonal_error_pct


def main():
    # 4: CSV laden
    df = pd.read_csv("dataset/labels.csv")
    df.head()
    # check:
    # print(df.shape)

    # 5: check a image
    # row = df.iloc[0]
    # img = Image.open(row["image_name"])
    # plt.imshow(img)
    # plt.show()
    # print(row["x"], row["y"])

    # 6: DataLoader
    # Transformationen:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.RandomAffine(
            degrees=3,
            translate=(0.02, 0.02)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset:
    train_dataset = GazeDataset(
        "./splits/subject_train.csv",
        transform, dataset_size=2000,
    )

    test_dataset = GazeDataset(
        "./splits/subject_test.csv",
        transform, dataset_size=500,
    )

    # print(len(train_dataset))
    # print(len(test_dataset))

    # Loader:
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        persistent_workers=True
    )

    # 7: ResNet18 (Pretrainiertes Modell):

    # Load:
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # last layer Original: 512 → 1000, but we need: 512 → 2 (for x, y)
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

    # 8: GPU

    # check:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # print(device)
    model.to(device)

    # 8: Loss and Optimizer

    # for regression:
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    for param in model.parameters():
        param.requires_grad = False

    # # Unfreeze the head
    for param in model.fc.parameters():
        param.requires_grad = True

    # Optimizer:
    optimizer = torch.optim.Adam(
        # model.parameters(),
        model.fc.parameters(),
        lr=1e-4
    )

    # 9: Training
    # epochs = 10
    # epochs = 2
    epochs = 15

    best_error, test_err_mean, test_err_max = None, None, None

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}"
        )

        for images, targets in loop:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(images)

            loss = criterion(
                preds,
                targets
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            loop.set_postfix(
                loss=loss.item()
            )

        # print(f"\ntrain_error:")
        # train_mae, train_rmse, train_diag_pct = diagonal_errors(model, train_loader, device)

        # print(f"MAE : {train_mae:.4f}")
        # print(f"RMSE: {train_rmse:.4f}")

        print(f"\ntest_error:")
        test_mae, test_rmse, test_diag_pct = diagonal_errors(model, test_loader, device)

        # print(f"MAE : {test_mae:.4f}")
        # print(f"RMSE: {test_rmse:.4f}")

        if best_error is None or test_rmse < best_error:
            best_error = test_rmse
            torch.save(model.state_dict(), "best_model.path")

        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1}: "
            f"{running_loss / len(train_loader):.4f} | "
            f"test_diag_error={test_diag_pct:.3f}% | "
            # f"train_diag_error={train_diag_pct:.3f}% | \n"
        )

        torch.save(model.state_dict(), "last_model.path")


if __name__ == "__main__":
    main()

"""
# save the model:
torch.save(
    model.state_dict(),
    "gaze_model.pth"
)


# Inference
model.load_state_dict(
    torch.load(
        "gaze_model.pth"
    )
)

model.eval()

predictions = []
targets_all = []

with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(device)
        preds = model(images)

        predictions.append(
            preds.cpu()
        )

        targets_all.append(
            targets
        )


predictions = torch.cat(
    predictions
)

targets_all = torch.cat(
    targets_all
)



# 10: Evaluation

screen_w = 984.0

mae = mean_absolute_error(
    targets_all.numpy(),
    predictions.numpy()
)

rmse = np.sqrt(
    mean_squared_error(
        targets_all.numpy(),
        predictions.numpy()
    )
)

print("MAE :", mae)
print("RMSE:", rmse)

#print(f"\nDiagonal-Error %: {100*np.round(mae / np.sqrt(2*screen_w**2), 3)} %")
print(f"\n\nDiagonal-Error %: {100*np.round(rmse / np.sqrt(2*screen_w**2), 3)} %")
"""