"""Data preparation: download, reformat, normalize, and split the GazeCapture dataset.

This script handles the full data pipeline:
1. Download GazeCapture from Kaggle (optional)
2. Reformat raw JSON/frames into labels.csv
3. Normalize coordinates to [0, 1] using screen resolution -> norm_labels.csv
4. Create train/test splits: random (group-shuffle) and subject-independent

Run without arguments to interactively create splits. Use main.py functions
programmatically in other scripts.
"""

import json
import csv
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit


# =============================================================================
# 1. Download / Reformat / Normalize
# =============================================================================

def reformat(in_path, out_path):
    """Convert raw GazeCapture frames + JSON annotations to labels.csv."""
    DATASET_ROOT = Path(in_path)
    OUTPUT_DIR = Path(out_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "labels.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "x", "y", "subject_ID"])

        total_images = 0
        total_subjects = 0

        for outer_subject_dir in sorted(DATASET_ROOT.iterdir()):
            if not outer_subject_dir.is_dir():
                continue

            subject_id = outer_subject_dir.name
            inner_subject_dir = outer_subject_dir / subject_id

            if not inner_subject_dir.exists():
                continue

            frames_file = inner_subject_dir / "frames.json"
            dotinfo_file = inner_subject_dir / "dotInfo.json"
            frames_folder = inner_subject_dir / "frames"

            if not frames_file.exists():
                print(f"Skipping {subject_id}: frames.json missing")
                continue
            if not dotinfo_file.exists():
                print(f"Skipping {subject_id}: dotInfo.json missing")
                continue
            if not frames_folder.exists():
                print(f"Skipping {subject_id}: frames folder missing")
                continue

            try:
                with open(frames_file, "r", encoding="utf-8") as f:
                    frame_names = json.load(f)
                with open(dotinfo_file, "r", encoding="utf-8") as f:
                    dot_info = json.load(f)
            except Exception as e:
                print(f"Skipping {subject_id}: JSON read error --> {e}")
                continue

            try:
                x_values = dot_info["XPts"]
                y_values = dot_info["YPts"]
            except KeyError as e:
                print(f"Skipping {subject_id}: missing key --> {e}")
                continue

            if len(frame_names) != len(x_values):
                print(f"ERROR in {subject_id}: {len(frame_names)} frames but {len(x_values)} labels")
                continue

            total_subjects += 1

            for idx in range(len(frame_names)):
                original_image_name = frame_names[idx]
                x = x_values[idx]
                y = y_values[idx]
                source_image = frames_folder / original_image_name

                if not source_image.exists():
                    print(f"Missing image: {source_image}")
                    continue

                writer.writerow([source_image, x, y, subject_id])
                total_images += 1

    print()
    print("=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Subjects processed : {total_subjects}")
    print(f"Images processed   : {total_images}")
    print(f"CSV saved to       : {csv_path}")


def normalize(in_path, out_path):
    """Normalize coordinates to [0, 1] using per-frame screen resolution."""
    DATASET_ROOT = Path(in_path)
    OUTPUT_DIR = Path(out_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "norm_labels.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "x", "y", "subject_ID", "screen_w", "screen_h"])

        total_images = 0
        total_subjects = 0

        for outer_subject_dir in sorted(DATASET_ROOT.iterdir()):
            if not outer_subject_dir.is_dir():
                continue

            subject_id = outer_subject_dir.name
            inner_subject_dir = outer_subject_dir / subject_id

            if not inner_subject_dir.exists():
                continue

            frames_file = inner_subject_dir / "frames.json"
            dotinfo_file = inner_subject_dir / "dotInfo.json"
            frames_folder = inner_subject_dir / "frames"
            screen_file = inner_subject_dir / "screen.json"

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
                print(f"Skipping {subject_id}: screen.json missing")
                continue

            try:
                with open(frames_file, "r", encoding="utf-8") as f:
                    frame_names = json.load(f)
                with open(dotinfo_file, "r", encoding="utf-8") as f:
                    dot_info = json.load(f)
                with open(screen_file, "r", encoding="utf-8") as f:
                    screen_info = json.load(f)
            except Exception as e:
                print(f"Skipping {subject_id}: JSON read error --> {e}")
                continue

            try:
                x_values = dot_info["XPts"]
                y_values = dot_info["YPts"]
                h_values = screen_info["H"]
                w_values = screen_info["W"]
            except KeyError as e:
                print(f"Skipping {subject_id}: missing key --> {e}")
                continue

            if len(frame_names) != len(x_values):
                print(f"ERROR in {subject_id}: {len(frame_names)} frames but {len(x_values)} labels")
                continue

            total_subjects += 1

            for idx in range(len(frame_names)):
                original_image_name = frame_names[idx]
                x = x_values[idx]
                y = y_values[idx]
                screen_h = h_values[idx]
                screen_w = w_values[idx]

                # Normalization to [0, 1]:
                x_norm = min(1, max(0, (x / screen_w)))
                y_norm = min(1, max(0, (y / screen_h)))

                source_image = frames_folder / original_image_name
                if not source_image.exists():
                    print(f"Missing image: {source_image}")
                    continue

                writer.writerow([source_image, x_norm, y_norm, subject_id, screen_w, screen_h])
                total_images += 1

    print()
    print("=" * 50)
    print(f"{'#'*10} Normalisation DONE {'#'*10}")
    print("=" * 100)
    print(f"Subjects processed : {total_subjects}")
    print(f"Images processed   : {total_images}")
    print(f"CSV saved to       : {csv_path}")


def download_kaggle():
    """Download GazeCapture dataset from Kaggle (requires API credentials)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print("\n\n******** Downloading ...! ********")
    api.dataset_download_files(
        'dhruv413/gaze-capture-20gb-zip',
        path='./dataset/images/gaze-capture-20gb',
        unzip=True
    )
    print("Download completed!")


# =============================================================================
# 2. Split Creation
# =============================================================================

def random_split(input_file):
    """Group-shuffle split: keeps all frames of a (subject, x, y) group together."""
    df = pd.read_csv(input_file)

    df["group_id"] = (
        df["subject_ID"].astype(str) + "_" +
        df["x"].astype(str) + "_" +
        df["y"].astype(str)
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["group_id"]))

    rand_train_df = df.iloc[train_idx].drop(columns=["group_id"])
    rand_test_df = df.iloc[test_idx].drop(columns=["group_id"])

    mapping = {
        "labels.csv": ("./splits/random_train.csv", "./splits/random_test.csv"),
        "norm_labels.csv": ("./splits/norm_random_train.csv", "./splits/norm_random_test.csv"),
    }

    filename = os.path.basename(input_file)
    train_name, test_name = mapping[filename]

    rand_train_df.to_csv(train_name, index=False)
    print(f"The file '{os.path.basename(train_name)}' is created! --> Train samples: {len(rand_train_df)}")

    rand_test_df.to_csv(test_name, index=False)
    print(f"The file '{os.path.basename(test_name)}' is created! --> Test samples: {len(rand_test_df)}\n")


def subjects_split(input_file):
    """Subject-independent split: disjoint subject IDs in train vs test."""
    df = pd.read_csv(input_file)

    subjects = df["subject_ID"].unique()
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=0.2, random_state=42, shuffle=True
    )
    sub_train_df = df[df["subject_ID"].isin(train_subjects)]
    sub_test_df = df[df["subject_ID"].isin(test_subjects)]

    mapping = {
        "norm_labels.csv": ("./splits/norm_subject_train.csv", "./splits/norm_subject_test.csv"),
        "labels.csv": ("./splits/subject_train.csv", "./splits/subject_test.csv"),
    }

    filename = os.path.basename(input_file)
    train_name, test_name = mapping[filename]

    sub_train_df.to_csv(train_name, index=False)
    print(f"The file '{os.path.basename(train_name)}' is created! --> Train samples: {len(sub_train_df)}")

    sub_test_df.to_csv(test_name, index=False)
    print(f"The file '{os.path.basename(test_name)}' is created! --> Test samples: {len(sub_test_df)}\n")


# =============================================================================
# 3. Split Validation
# =============================================================================

def check_split(input_train, input_test):
    """Verify that train/test splits have no subject overlap."""
    train_df = pd.read_csv(input_train)
    test_df = pd.read_csv(input_test)

    train_subjects = set(train_df["subject_ID"].unique())
    test_subjects = set(test_df["subject_ID"].unique())

    overlap = train_subjects.intersection(test_subjects)

    if len(overlap) == 0:
        print("\n************\nOk, No overlapping subjects between train and test!\n************\n")
    else:
        print("\n************\nERROR: Overlapping subjects detected!\n************\n")
        print(f"\nNumber of Overlapping subjects: {len(overlap)}\n")
        print(f"\n\nExample overlap:\n", list(overlap)[:10])

    all_subjects = train_subjects.union(test_subjects)
    print(f"Total unique subjects in split: {len(all_subjects)}")

    train_counts = train_df["subject_ID"].value_counts()
    test_counts = test_df["subject_ID"].value_counts()
    common_rows = set(train_counts.index).intersection(set(test_counts.index))

    if common_rows:
        print("\n************\nWARNING: These subjects appear in both splits:")
        print(list(common_rows)[:10])
    else:
        print("\n************\nOK, No subject appears in both splits at row level!!\n************\n")


# =============================================================================
# 4. Main Pipeline
# =============================================================================

def main():
    """Interactive pipeline: download -> reformat -> normalize -> split."""
    dataset_path = './dataset/images/'

    # 1. Check / download dataset
    if not os.path.exists('./dataset/images/00002/00002/frames/00000.jpg'):
        print("\nDataset not found!")
        confirm = input("\nDownload the dataset from Kaggle? (y/n): ")
        if confirm.lower() == 'y':
            download_kaggle()
    else:
        print(f"Dataset found: {dataset_path}")

    # 2. Create labels.csv if missing
    if not os.path.exists('./dataset/labels.csv'):
        print(f"\n'labels.csv' not found. Creating from raw data...")
        reformat(dataset_path, "./dataset/")

    # 3. Create norm_labels.csv if missing
    if not os.path.exists('./dataset/norm_labels.csv'):
        print(f"\n'norm_labels.csv' not found. Creating normalized labels...")
        normalize(dataset_path, "./dataset/")

    # 4. Choose which CSV to split
    while True:
        confirm_split = input("\n\nCreate split-files for 'labels.csv' or 'norm_labels.csv'? (l/n): ")
        if confirm_split.lower() == 'l':
            csv_path = './dataset/labels.csv'
            break
        elif confirm_split.lower() == 'n':
            csv_path = './dataset/norm_labels.csv'
            break
        else:
            print("Invalid input. Please enter 'l' or 'n'.")

    # 5. Create random split if missing
    if not os.path.exists('./splits/random_train.csv') or not os.path.exists('./splits/random_test.csv'):
        random_split(csv_path)

    # 6. Create subject-independent split if missing
    if not os.path.exists('./splits/subject_train.csv') or not os.path.exists('./splits/subject_test.csv'):
        subjects_split(csv_path)

    # 7. Optional validation
    split_test = input("\nValidate splits? (y/n): ")
    if split_test.lower() == 'y':
        which = input("Validate which split? (n=norm_subject / r=norm_random): ")
        if which.lower() == 'n':
            check_split("./splits/norm_subject_train.csv", "./splits/norm_subject_test.csv")
        elif which.lower() == 'r':
            check_split("./splits/norm_random_train.csv", "./splits/norm_random_test.csv")
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()