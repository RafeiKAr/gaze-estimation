"""Dataset loading and preprocessing for the gaze-estimation model.

Reads the pre-computed split CSVs under ./splits (produced by main.py) and
exposes a torch Dataset plus a helper to build train/test DataLoaders.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SPLITS_DIR = Path("./splits")

# split_type -> (train_csv, test_csv), both normalized to [0, 1] targets.
SPLIT_FILES = {
    "norm_subject": (SPLITS_DIR / "norm_subject_train.csv", SPLITS_DIR / "norm_subject_test.csv"),
    "norm_random": (SPLITS_DIR / "norm_random_train.csv", SPLITS_DIR / "norm_random_test.csv"),
}


def build_transform():
    """Image preprocessing/augmentation used for both train and test splits."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class GazeDataset(Dataset):
    """Loads (image, normalized (x, y) target) pairs from a labels CSV.

    Images are decoded and transformed once at construction time and kept
    in memory as a single tensor, since personalization-scale datasets are
    small enough for this to be fast to iterate over during training.
    """

    def __init__(self, csv_file, transform, dataset_size=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.dataset_size = dataset_size if dataset_size is not None else len(self.df)

        probe = transform(Image.new("RGB", (500, 300)))
        self.images = torch.zeros([self.dataset_size] + list(probe.shape))
        self.targets = torch.zeros(self.dataset_size, 2)

        for idx in range(self.dataset_size):
            row = self.df.iloc[idx]
            image = Image.open(row["image_name"]).convert("RGB")
            self.images[idx] = self.transform(image)
            self.targets[idx] = torch.tensor([row["x"], row["y"]], dtype=torch.float32)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def build_dataloaders(
    split_type,
    train_size,
    test_size,
    batch_size,
    num_workers=8,
):
    """Build train/test DataLoaders for one of SPLIT_FILES.

    Args:
        split_type: "norm_subject" or "norm_random".
        train_size: number of training samples to load (subset of the split).
        test_size: number of test samples to load.
        batch_size: batch size shared by both loaders.
        num_workers: DataLoader worker processes.
    """
    if split_type not in SPLIT_FILES:
        raise ValueError(f"Unknown split_type: {split_type!r}, expected one of {list(SPLIT_FILES)}")

    train_csv, test_csv = SPLIT_FILES[split_type]
    transform = build_transform()

    train_dataset = GazeDataset(train_csv, transform, dataset_size=train_size)
    test_dataset = GazeDataset(test_csv, transform, dataset_size=test_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def load_normalized_targets(csv_path="./dataset/norm_labels.csv"):
    """Load the full (x, y) normalized target array, e.g. for baselines."""
    df = pd.read_csv(csv_path)
    return df[["x", "y"]].to_numpy(dtype=np.float64)


class EmbeddingDataset(Dataset):
    """Wraps precomputed (backbone embedding, target) pairs."""

    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


@torch.no_grad()
def compute_embeddings(csv_file, transform, backbone, device, dataset_size=None, batch_size=64, cache_path=None):
    """Run a frozen backbone once over csv_file and return (embeddings, targets).

    Only valid when the backbone is fully frozen (trainable_layers ==
    "fc_only"): the embeddings are otherwise stale as soon as the backbone
    is updated. If cache_path is given and already exists, it is loaded
    instead of recomputing -- this is what lets sweeps that vary only the
    head (batch size, learning rate, activation, loss function) skip the
    ResNet forward pass entirely after the first run.
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = torch.load(cache_path)
        return cached["embeddings"], cached["targets"]

    image_dataset = GazeDataset(csv_file, transform, dataset_size=dataset_size)
    loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    backbone.eval()
    backbone.to(device)

    embeddings_list, targets_list = [], []
    for images, targets in loader:
        embeddings_list.append(backbone(images.to(device)).cpu())
        targets_list.append(targets)

    embeddings = torch.cat(embeddings_list)
    targets = torch.cat(targets_list)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings, "targets": targets}, cache_path)

    return embeddings, targets


def build_embedding_dataloaders(
    split_type,
    train_size,
    test_size,
    batch_size,
    backbone,
    device,
    cache_dir=None,
    extraction_batch_size=64,
):
    """Like build_dataloaders, but yields precomputed backbone embeddings
    instead of images, for training/evaluating a head only (model.build_head)
    while the backbone stays frozen.

    If cache_dir is given, embeddings are cached to
    "{cache_dir}/{split_type}_{train|test}_{size}.pt" and reused across
    calls, e.g. across a hyperparameter sweep that only varies the head.
    """
    if split_type not in SPLIT_FILES:
        raise ValueError(f"Unknown split_type: {split_type!r}, expected one of {list(SPLIT_FILES)}")

    train_csv, test_csv = SPLIT_FILES[split_type]
    transform = build_transform()

    train_cache = Path(cache_dir) / f"{split_type}_train_{train_size}.pt" if cache_dir else None
    test_cache = Path(cache_dir) / f"{split_type}_test_{test_size}.pt" if cache_dir else None

    train_embeddings, train_targets = compute_embeddings(
        train_csv, transform, backbone, device, train_size, extraction_batch_size, train_cache
    )
    test_embeddings, test_targets = compute_embeddings(
        test_csv, transform, backbone, device, test_size, extraction_batch_size, test_cache
    )

    train_loader = DataLoader(
        EmbeddingDataset(train_embeddings, train_targets), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        EmbeddingDataset(test_embeddings, test_targets), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader
