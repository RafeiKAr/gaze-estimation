
# 1: Bib

import time
start_time = time.perf_counter()

import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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

from torch.utils.tensorboard import SummaryWriter


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

    mae = mean_absolute_error(targets_all, predictions)

    rmse = np.sqrt(mean_squared_error(targets_all, predictions))

    diagonal_error_pct = np.round(100 * (rmse / np.sqrt(2)), 5)

    print(f"MAE : {mae:.4f} \t RMSE: {rmse:.4f} ")
    # print(f"RMSE: {rmse:.4f}")
    print(f"Diagonal-Error %: {diagonal_error_pct:.4f} %")

    return mae, rmse, diagonal_error_pct


def main():

    # 3: Hyperparameter
    # def_dataset_size = [1000, 200]
    def_dataset_size = [10000, 2000]
    
    optimizer_name = "AdamW"
    batch_Size = 64

    epochs = 10
    # epochs = 2

    learning_rate = 1e-4
    weight_Decay = 1e-4
    

    # 6: CSV laden
    # df = pd.read_csv("dataset/labels.csv")
    df = pd.read_csv("dataset/norm_labels.csv")
    df.head()
    # check:
    # print(df.shape)


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

    # def_dataset = 'norm_subject'
    #
    # train_dataset = GazeDataset(
    #     "./splits/norm_subject_train.csv",
    #     transform, dataset_size=def_dataset_size[0],
    # )
    #
    # test_dataset = GazeDataset(
    #     "./splits/norm_subject_test.csv",
    #     transform, dataset_size=def_dataset_size[1],
    # )

    def_dataset = 'norm_random'

    train_dataset = GazeDataset(
        "./splits/norm_random_train.csv",
        transform, dataset_size=def_dataset_size[0],
    )

    test_dataset = GazeDataset(
        "./splits/norm_random_test.csv",
        transform, dataset_size=def_dataset_size[1],
    )

    # Loader:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_Size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_Size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True
    )


    # 9: ResNet18 (Pretrainiertes Modell):
    # Load:
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model_name = model.__class__.__name__

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
        lr=learning_rate,
        weight_decay=weight_Decay
    )


    # 12: Training
    epochs = epochs

    model_output = './models'
    model_dir = Path(model_output)

    if not os.path.exists(model_dir):
        model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    best_error = None
    diag_test_error, diag_train_error = [], []

    train_start = time.perf_counter()

    writer = SummaryWriter(
        log_dir=os.path.join("runs", f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                                     f"{def_dataset}_{def_dataset_size[0]}-{def_dataset_size[1]}")
    )

    for epoch in range(epochs):

        epoch_start = time.perf_counter()

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

        # epoch_end = time.perf_counter()

        print(f"\ntrain_error:")
        train_mae, train_rmse, train_diag_pct = diagonal_errors(model, train_loader, device)
        diag_train_error.append(np.round(train_diag_pct, 4))

        print(f"\ntest_error:")
        test_mae, test_rmse, test_diag_pct = diagonal_errors(model, test_loader, device)
        diag_test_error.append(np.round(test_diag_pct, 4))

        # writer.add_scalar("train_error/epoch", float(f"{train_diag_pct:.4f}"), epoch + 1)
        # writer.add_scalar("test_error/epoch", float(f"{test_diag_pct:.4f}"), epoch + 1)
        # writer.add_scalar("train_loss/epoch", running_loss / len(train_loader), epoch + 1)
        writer.add_scalars(
            main_tag="Diagnostic Error",
            tag_scalar_dict={
                "Train": f"{train_diag_pct:.4f}",
                "Test": f"{test_diag_pct:.4f}",
                "Constant": 27.69,
                "Random": 45.53
            },
            global_step=epoch + 1
        )
        writer.flush()

        if best_error is None or test_rmse < best_error:
            best_error = test_rmse
            torch.save(model.state_dict(),
                    f"./models/{model_name}_best-model_{def_dataset}_{def_dataset_size[0]}-{def_dataset_size[1]}.path")

        epoch_end = time.perf_counter()


        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1}: {epoch_end - epoch_start:.2f} Sekunden | "
            f"Running_loss: {running_loss / len(train_loader):.3f} | "
            f"test_diag_error={test_diag_pct:.4f}% | "
            f"train_diag_error={train_diag_pct:.4f}% | \n"
        )

        torch.save(model.state_dict(),
                   f"./models/{model_name}_last-model_{def_dataset}_{def_dataset_size[0]}-{def_dataset_size[1]}.path")


    train_end = time.perf_counter()
    end_time = time.perf_counter()

    writer.add_hparams(
        {
            "dataset": def_dataset,
            "dataset_size": str(def_dataset_size),
            "optimizer": optimizer_name,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_Decay),
            "batch_size": batch_Size,
            "epochs": epochs
        },
        {
            "hparam/test_diag_error": float(diag_test_error[-1]),
            "hparam/train_diag_error": float(diag_train_error[-1]),
            "hparam/total_running_time_min": float((end_time - start_time) / 60),
            "hparam/train_running_time_min": float((train_end - train_start) / 60)
        }
    )

    writer.add_text(
        tag=f"Train-Error & Test-Error for {epochs} Epochs",
        text_string=f"Train-Error: {np.float64(diag_train_error)}\n"
                    f"Test-Error: {np.float64(diag_test_error)}",
        global_step=0
    )

    writer.flush()



    # train_end = time.perf_counter()
    elapsed_running_time = train_end - train_start
    print(f"\ntotll running-tiems (s): {elapsed_running_time:.2f} Sekunden")
    print(f"totll running-tiems (min): {elapsed_running_time / 60:.2f} Minuten\n")


    # end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nGesamte Laufzeit: {elapsed_time:.2f} Sekunden")
    print(f"Gesamte Laufzeit: {elapsed_time / 60:.2f} Minuten\n")



    print(f"\n\n Diagonal-Train-Error: {np.float64(diag_train_error)}\n")
    print(f" Diagonal-Test-Error: {np.float64(diag_test_error)}\n\n")

    print(
        f" Diagonal-Train-Error-Max: {np.max(diag_train_error)} \t Diagonal-Train-Error-Min: {np.min(diag_train_error)}\t")
    # print(f" Diagonal-Train-Error-Ave: {np.mean(diag_train_error)}\n")

    print(
        f" Diagonal-Test-Error-Max: {np.max(diag_test_error)}\t Diagonal-Test-Error-Min: {np.min(diag_test_error)}\t ")
    # print(f" Diagonal-Test-Error-Ave: {np.mean(diag_test_error)}\n")

    # Curven:
    epochs = range(0, len(diag_test_error) + 1)
    # epochs = range(1, len(diag_train_error) + 1)


    plt.figure(figsize=(8, 5))

    plt.plot(diag_train_error, label="Train", linewidth=2)
    plt.plot(diag_test_error, label="Test", linewidth=2)

    plt.plot(epochs, [27.245] * len(epochs),
         label="Constant Error", linewidth=1)

    plt.plot(epochs, [38.939] * len(epochs),
         label="Random Error", linewidth=1)

    plt.title(f"{def_dataset} | (train: {def_dataset_size[0]}, test: {def_dataset_size[1]}) |"
              f" lr = {learning_rate} | {elapsed_running_time / 60:.2f} Minuten"
              )

    plt.xlabel("Epoche")
    plt.ylabel("Diagnostic Error (%)")
    plt.grid(True)
    plt.legend()

    # Ordner erstellen (falls er noch nicht existiert)
    os.makedirs("./results", exist_ok=True)

    # Diagramm speichern
    plt.savefig(
        f"results/diag_error_{def_dataset}_{learning_rate}_{def_dataset_size[0]}_{def_dataset_size[1]}_a.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    writer.close()

    



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