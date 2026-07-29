# 1: Bib
import os
import re
import json
import csv
from pathlib import Path
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


# 3: Normalization:
def normalize(in_path, out_path):

    DATASET_ROOT = Path(in_path)

    OUTPUT_DIR = Path(out_path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGES_DIR = OUTPUT_DIR / "images"

    csv_path = OUTPUT_DIR / "norm_labels.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "image_name",
            "x_norm",
            "y_norm",
            "subject_ID",
            "screen_w",
            "screen_h"
        ])

        total_images = 0
        total_subjects = 0

        # ======================================================
        # Alle Subjekte durchlaufen
        # ======================================================

        for outer_subject_dir in sorted(DATASET_ROOT.iterdir()):

            if not outer_subject_dir.is_dir():
                continue

            subject_id = outer_subject_dir.name

            inner_subject_dir = (
                outer_subject_dir / subject_id
            )

            if not inner_subject_dir.exists():
                continue

            # ==================================================
            # Dateien laden
            # ==================================================

            frames_file = (
                inner_subject_dir / "frames.json"
            )

            dotinfo_file = (
                inner_subject_dir / "dotInfo.json"
            )

            frames_folder = (
                inner_subject_dir / "frames"
            )

            screen_file = (
                    inner_subject_dir / "screen.json"
            )

            # --------------------------------------------------
            if not frames_file.exists():
                print(f"Skipping {subject_id}: frames.json missing")
                continue

            if not dotinfo_file.exists():
                print(f"Skipping {subject_id}: dotInfo.json missing")
                continue

            if not frames_folder.exists():
                print(f"Skipping {subject_id}: frames folder missing")
                continue

            if not screen_file.exists():
                print(f"Skipping {subject_id}: frames folder missing")
                continue
            # --------------------------------------------------

            try:
                with open(frames_file, "r", encoding="utf-8") as f:
                    frame_names = json.load(f)

                with open(dotinfo_file, "r", encoding="utf-8") as f:
                    dot_info = json.load(f)

                with open(screen_file, "r", encoding="utf-8") as f:
                    screen_info = json.load(f)

            except Exception as e:
                print(f"Skipping {subject_id}: JSON read error --> {e}")

            try:
                x_values = dot_info["XPts"]
                y_values = dot_info["YPts"]

                h_values = screen_info["H"]
                w_values = screen_info["W"]

            except KeyError as e:
                print(f"Skipping {subject_id}: missing key --> {e}")

            if len(frame_names) != len(x_values):
                print(
                    f"ERROR in {subject_id}: "
                    f"{len(frame_names)} frames but "
                    f"{len(x_values)} labels"
                )
                continue

            total_subjects += 1

            # ==================================================
            # Alle Frames bearbeiten
            # ==================================================

            for idx in range(len(frame_names)):

                original_image_name = frame_names[idx]

                x = x_values[idx]
                y = y_values[idx]

                screen_h = h_values[idx]
                screen_w = w_values[idx]


                # Normalisation:
                x_norm = min(1, max(0, (x / screen_w)))
                y_norm = min(1, max(0, (y / screen_h)))



                source_image = (
                    frames_folder /
                    original_image_name
                )

                if not source_image.exists():
                    print(
                        f"Missing image: {source_image}"
                    )
                    continue


                writer.writerow([
                    source_image,
                    x_norm,
                    y_norm,
                    subject_id,
                    screen_w,
                    screen_h
                ])

                total_images += 1


    print()
    print("=" * 50)
    print(f"{'#'*10} Normalisation DONE {'#'*10}")
    print("=" * 100)
    print(f"Subjects processed : {total_subjects}")
    print(f"Images processed   : {total_images}")
    print(f"CSV saved to       : {csv_path}")



# 4: func-Screen-size:
def find_screen_size():
    # Checking for normalize-file (norm_labels.csv):
    dataset_path = './dataset/images/'
    if not os.path.exists('./dataset/norm_labels.csv'):

        print(f"The file: 'norm_labels.csv' is maybe needed, but doesn't find!\n Try to create it ... \n")
        normalize(dataset_path, "./dataset/")

        if os.path.exists('./dataset/norm_labels.csv'):
            print(f"Now can find the file: 'norm_labels.csv' !\n ")

    """Suche die Bildschirmbreite aus einer Datei mit screen_w und screen_h"""
    candidates = [
        # "norm_labels.py",
        # "dataset/labels.csv",
        "dataset/norm_labels.csv"
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

    raise ValueError("screen_w konnte nicht gefunden werden. Bitte Datei mit 'screen_w' und 'screen_h' angeben.")


# 5: func-Diagonal-Error:
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

    # screen_w, screen_h = find_screen_size()

    mae = mean_absolute_error(targets_all, predictions)

    rmse = np.sqrt(mean_squared_error(targets_all, predictions))

    # diagonal_pixels = np.sqrt(screen_w**2 + screen_h**2)
    # diagonal_error_pct = np.round(rmse / diagonal_pixels, 3) * 100

    diagonal_error_pct = np.round(100 * (rmse / np.sqrt(2)), 5)

    print(f"MAE : {mae:.4f} \t RMSE: {rmse:.4f} ")
    # print(f"RMSE: {rmse:.4f}")
    print(f"Diagonal-Error %: {diagonal_error_pct:.4f} %")

    return mae, rmse, diagonal_error_pct


def main():

    # 6: CSV laden
    df = pd.read_csv("dataset/labels.csv")
    df.head()
    # check:
    # print(df.shape)


    # 7: check a image
    # row = df.iloc[0]
    # img = Image.open(row["image_name"])
    # plt.imshow(img)
    # plt.show()
    # print(row["x"], row["y"])


    # 8: DataLoader
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

    # Dataset:
    train_dataset = GazeDataset(
        "./splits/norm_subject_train.csv",
        transform, dataset_size=16000,
    )

    test_dataset = GazeDataset(
        "./splits/norm_subject_test.csv",
        transform, dataset_size=4000,
    )

    # train_dataset = GazeDataset(
    #     "./splits/norm_random_train.csv",
    #     transform, dataset_size=2000,
    # )

    # test_dataset = GazeDataset(
    #     "./splits/norm_random_test.csv",
    #     transform, dataset_size=500,
    # )

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


    # 9: ResNet18 (Pretrainiertes Modell):
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


    # 10: GPU
    # check:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # print(device)
    model.to(device)


    # 11: Loss and Optimizer
    # for regression:
    # criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    #
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # # # Unfreeze the head
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    #
    # # Optimizer:
    # optimizer = torch.optim.Adam(
    #     # model.parameters(),
    #     model.fc.parameters(),
    #     lr=1e-5
    # )

    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the head
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,
        weight_decay=1e-5
    )


    # 12: Training
    epochs = 10
    # epochs = 2

    model_output = './models'
    model_dir = Path(model_output)

    if not os.path.exists(model_dir):
        model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    best_error = None
    diag_test_error, diag_train_error = [], []

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

        print(f"\ntrain_error:")
        train_mae, train_rmse, train_diag_pct = diagonal_errors(model, train_loader, device)
        diag_train_error.append(np.round(train_diag_pct, 4))

        print(f"\ntest_error:")
        test_mae, test_rmse, test_diag_pct = diagonal_errors(model, test_loader, device)
        diag_test_error.append(np.round(test_diag_pct, 4))

        if best_error is None or test_rmse < best_error:
            best_error = test_rmse
            torch.save(model.state_dict(), "./models/best_model.path")

        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1}: "
            f"Running_loss: {running_loss / len(train_loader):.3f} | "
            f"test_diag_error={test_diag_pct:.4f}% | "
            f"train_diag_error={train_diag_pct:.4f}% | \n"
        )

        torch.save(model.state_dict(), "./models/last_model.path")

    # print(f"\n\nDiagonal_train_error: {np.float64(diag_train_error)}")
    # print(f"\n\nDiagonal_test_error: {np.float64(diag_test_error)}")

    print(f"\n\n Diagonal-Train-Error: {np.float64(diag_train_error)}\n")
    print(f" Diagonal-Test-Error: {np.float64(diag_test_error)}\n\n")

    print(f" Diagonal-Train-Error-Max: {np.max(diag_train_error)} \t Diagonal-Train-Error-Min: {np.min(diag_train_error)}\t")
    print(f" Diagonal-Train-Error-Ave: {np.round(np.mean(diag_train_error), 4)}\n")

    print(f" Diagonal-Test-Error-Max: {np.max(diag_test_error)}\t Diagonal-Test-Error-Min: {np.min(diag_test_error)}\t ")
    print(f" Diagonal-Test-Error-Ave: {np.round(np.mean(diag_test_error), 4)}\n")


    # Curven:

    epochs = range(1, len(diag_test_error) + 1)
    # epochs = range(1, len(diag_train_error) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(diag_train_error, label="Train", linewidth=2)
    plt.plot(diag_test_error, label="Test", linewidth=2)

    plt.title("Diagnostic Error über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("Diagnostic Error (%)")
    plt.grid(True)
    plt.legend()

    plt.show()

    



if __name__ == "__main__":
    main()







"""
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