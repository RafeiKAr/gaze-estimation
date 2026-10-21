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

    def __init__(self, csv_file, transform, dataset_size=None, indices=None):
        """
        Args:
            indices: explicit row indices to load (see select_rows). If None,
                the FIRST `dataset_size` rows are used (legacy behaviour; the
                split CSVs are ordered by subject, so e.g. the first 1000
                training rows come from only 5 subjects).
        """
        df = pd.read_csv(csv_file) if not isinstance(csv_file, pd.DataFrame) else csv_file
        if indices is not None:
            df = df.iloc[np.asarray(indices)].reset_index(drop=True)
        self.df = df
        self.transform = transform
        self.dataset_size = min(dataset_size, len(self.df)) if dataset_size is not None else len(self.df)

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


def select_rows(df, n, seed=None, exclude_subjects=None):
    """Row indices of a subset of size n.

    seed None  -> first n rows (legacy, reproduces all historical results)
    seed int   -> n rows drawn uniformly at random (reproducible), optionally
                  excluding whole subjects (for subject-disjoint validation).
    """
    pool = np.arange(len(df))
    if exclude_subjects is not None and "subject_ID" in df.columns:
        pool = pool[~df["subject_ID"].isin(set(exclude_subjects)).to_numpy()]
    if seed is None:
        return pool[:n]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))


def select_validation_subjects(df, val_size, seed):
    """Pick whole subjects from the training CSV until >= val_size rows."""
    rng = np.random.default_rng(seed)
    subjects = df["subject_ID"].unique()
    rng.shuffle(subjects)
    counts = df["subject_ID"].value_counts()
    chosen, total = [], 0
    for subj in subjects:
        chosen.append(subj)
        total += counts[subj]
        if total >= val_size:
            break
    return chosen


def build_dataloaders(
    split_type,
    train_size,
    test_size,
    batch_size,
    num_workers=8,
    sample_seed=None,
    val_size=0,
):
    """Build train/test DataLoaders for one of SPLIT_FILES.

    Args:
        split_type: "norm_subject" or "norm_random".
        train_size: number of training samples to load (subset of the split).
        test_size: number of test samples to load.
        batch_size: batch size shared by both loaders.
        num_workers: DataLoader worker processes.
        sample_seed: None = first-N rows (legacy); int = random subset.
        val_size: if > 0, a subject-disjoint validation set is drawn from the
            training CSV (requires sample_seed) and a third loader returned.

    Returns (train_loader, test_loader) or, if val_size > 0,
    (train_loader, test_loader, val_loader).
    """
    if split_type not in SPLIT_FILES:
        raise ValueError(f"Unknown split_type: {split_type!r}, expected one of {list(SPLIT_FILES)}")

    train_csv, test_csv = SPLIT_FILES[split_type]
    transform = build_transform()
    train_df, test_df = pd.read_csv(train_csv), pd.read_csv(test_csv)

    val_subjects = None
    if val_size > 0:
        if sample_seed is None:
            raise ValueError("val_size > 0 requires a sample_seed")
        val_subjects = select_validation_subjects(train_df, val_size, sample_seed)
    train_idx = select_rows(train_df, train_size, sample_seed, exclude_subjects=val_subjects)
    test_idx = select_rows(test_df, test_size, sample_seed)

    train_dataset = GazeDataset(train_df, transform, indices=train_idx)
    test_dataset = GazeDataset(test_df, transform, indices=test_idx)

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
    if val_size > 0:
        val_idx = np.flatnonzero(train_df["subject_ID"].isin(set(val_subjects)).to_numpy())[:val_size]
        val_loader = DataLoader(
            GazeDataset(train_df, transform, indices=val_idx),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        return train_loader, test_loader, val_loader
    return train_loader, test_loader


def subset_targets(split_type, train_size, test_size, sample_seed=None):
    """(train_targets, test_targets) of exactly the subset a loader would use.

    Needed to compute constant/random baselines on the same data as the model
    (see metrics.subset_baselines) instead of on the full dataset.
    """
    train_csv, test_csv = SPLIT_FILES[split_type]
    train_df, test_df = pd.read_csv(train_csv), pd.read_csv(test_csv)
    tr = train_df.iloc[select_rows(train_df, train_size, sample_seed)][["x", "y"]].to_numpy(np.float64)
    te = test_df.iloc[select_rows(test_df, test_size, sample_seed)][["x", "y"]].to_numpy(np.float64)
    return tr, te


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
def compute_embeddings(csv_file, transform, backbone, device, dataset_size=None, batch_size=64, cache_path=None, indices=None):
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

    image_dataset = GazeDataset(csv_file, transform, dataset_size=dataset_size, indices=indices)
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
    sample_seed=None,
    val_size=0,
):
    """Like build_dataloaders, but yields precomputed backbone embeddings
    instead of images, for training/evaluating a head only (model.build_head)
    while the backbone stays frozen.

    If cache_dir is given, embeddings are cached to
    "{cache_dir}/{split_type}_{part}_{size}[_s{seed}].pt" and reused across
    calls, e.g. across a hyperparameter sweep that only varies the head.
    Returns (train, test) or (train, test, val) loaders like build_dataloaders.
    """
    if split_type not in SPLIT_FILES:
        raise ValueError(f"Unknown split_type: {split_type!r}, expected one of {list(SPLIT_FILES)}")

    train_csv, test_csv = SPLIT_FILES[split_type]
    transform = build_transform()
    train_df, test_df = pd.read_csv(train_csv), pd.read_csv(test_csv)

    val_subjects = None
    if val_size > 0:
        if sample_seed is None:
            raise ValueError("val_size > 0 requires a sample_seed")
        val_subjects = select_validation_subjects(train_df, val_size, sample_seed)
    train_idx = select_rows(train_df, train_size, sample_seed, exclude_subjects=val_subjects)
    test_idx = select_rows(test_df, test_size, sample_seed)

    suffix = "" if sample_seed is None else f"_s{sample_seed}"
    if val_size > 0:
        suffix += f"_v{val_size}"

    def cache(part, size):
        return Path(cache_dir) / f"{split_type}_{part}_{size}{suffix}.pt" if cache_dir else None

    def embed(df, idx, part, size):
        return compute_embeddings(df, transform, backbone, device, None, extraction_batch_size,
                                  cache(part, size), indices=idx)

    train_embeddings, train_targets = embed(train_df, train_idx, "train", train_size)
    test_embeddings, test_targets = embed(test_df, test_idx, "test", test_size)

    train_loader = DataLoader(
        EmbeddingDataset(train_embeddings, train_targets), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        EmbeddingDataset(test_embeddings, test_targets), batch_size=batch_size, shuffle=False
    )
    if val_size > 0:
        val_idx = np.flatnonzero(train_df["subject_ID"].isin(set(val_subjects)).to_numpy())[:val_size]
        val_embeddings, val_targets = embed(train_df, val_idx, "val", val_size)
        val_loader = DataLoader(
            EmbeddingDataset(val_embeddings, val_targets), batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader, val_loader
    return train_loader, test_loader
