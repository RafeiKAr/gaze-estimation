
# 1: Bib
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

    def __init__(self, csv_file, transform=None, dataset_size=None):

        self.df = pd.read_csv(csv_file)

        if dataset_size is not None:
            self.df = self.df.iloc[:dataset_size]

        self.transform = transform


    def __len__(self):
        return len(self.df)
        # return self.dataset_size

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_name"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(
            [row["x"], row["y"]],
            dtype=torch.float32
        )

        return image, target
        # return self.images[idx], self.targets[idx]


# 3: func-diagonla-error:
def diagonal_errors(model, loader, device):
    model.eval()

    errors = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            # Euclidean distance per sample
            dist = torch.sqrt(((preds - targets) ** 2).sum(dim=1))

            # normalize by image diagonal
            dist = dist / np.sqrt(2)

            errors.append(dist)

    errors = torch.cat(errors)

    return errors.mean().item(), errors.max().item()



def main():
    # 4: CSV laden
    df = pd.read_csv("dataset/norm_labels.csv")
    df.head()
    # check:
    #print(df.shape)

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

    # 7: Dataset:
    train_dataset = GazeDataset(
        "./splits/subject_train.csv",
        transform, dataset_size=10000,
    )
    test_dataset = GazeDataset(
        "./splits/subject_test.csv",
        transform, dataset_size=2500,
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

    # 8: ResNet18 (pretrainate Modell):
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

    # 9: GPU
    # check:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    # print(device)
    model.to(device)

    # 10: Loss and Optimizer

    # for regression:
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the head
    for param in model.fc.parameters():
        param.requires_grad = True

    # Optimizer:
    optimizer = torch.optim.Adam(
        # model.parameters(),
        model.fc.parameters(),
        lr = 1e-4
    )

    # 11: Training
    epochs = 10
    # epochs = 2
    # epochs = 10

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
        train_mean_err, train_max_err = diagonal_errors(model, train_loader, device)
        test_mean_err, test_max_err = diagonal_errors(model, test_loader, device)

        if best_error is None or test_mean_err < best_error:
            best_error = test_mean_err
            torch.save(model.state_dict(), "best_model.path")


        print(
            f"[{[datetime.now().strftime('%H:%M:%S')]}] Epoch {epoch + 1}: "
            f"{running_loss / len(train_loader):.4f} | "
            f"train_diag_max={100*train_max_err:.3f}% | test_diag_max={100*test_max_err:.3f}% \n"
        )

        torch.save(model.state_dict(), "last_model.pth")



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


"""