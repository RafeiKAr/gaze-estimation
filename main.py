
from kaggle.api.kaggle_api_extended import KaggleApi

import json
import csv
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

#from Abschloss.backup.dataset_reformat import skipped_subjects


def reformat(in_path, out_path):
    # shutil.unpack_archive(zip_path, extract_to)

    DATASET_ROOT = Path(in_path)

    OUTPUT_DIR = Path(out_path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGES_DIR = OUTPUT_DIR / "images"

    csv_path = OUTPUT_DIR / "labels.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "image_name",
            "x",
            "y",
            "subject_ID"
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

            # --------------------------------------------------
            if not frames_file.exists():
                print(f"Skipping {subject_id}: frames.json missing")
                # skipped_subjects += 1
                continue

            if not dotinfo_file.exists():
                print(f"Skipping {subject_id}: dotInfo.json missing")
                # skipped_subjects += 1
                continue

            if not frames_folder.exists():
                print(f"Skipping {subject_id}: frames folder missing")
                # skipped_subjects += 1
                continue

            # --------------------------------------------------

            try:
                with open(frames_file, "r", encoding="utf-8") as f:
                    frame_names = json.load(f)

                with open(dotinfo_file, "r", encoding="utf-8") as f:
                    dot_info = json.load(f)


            except Exception as e:
                print(f"Skipping {subject_id}: JSON read error --> {e}")

            try:
                x_values = dot_info["XPts"]
                y_values = dot_info["YPts"]

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
                    x,
                    y,
                    subject_id
                ])

                total_images += 1

    print()
    print("=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Subjects processed\t: {total_subjects}")
    print(f"Images processed\t: {total_images}")
    print(f"CSV saved to\t\t: {csv_path}")


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
            "x",
            "y",
            "subject_ID"
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

                h = h_values[idx]
                w = w_values[idx]


                # Normalisation:
                x_norm = x / w
                y_norm = y / h


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
                    subject_id
                ])

                total_images += 1


    print()
    print("=" * 50)
    print(f"{'#'*10} Normalisation DONE {'#'*10}")
    print("=" * 100)
    print(f"Subjects processed : {total_subjects}")
    print(f"Images processed   : {total_images}")
    print(f"CSV saved to       : {csv_path}")


def download_kaggle():
    api = KaggleApi()
    api.authenticate()

    print("\n\n******** Dowanloading ...! ********")

    api.dataset_download_files(
        'dhruv413/gaze-capture-20gb-zip',
        path='./dataset/images/gaze-capture-20gb',
        unzip=True
    )

    print("Download completed!")


def random_split(input_file):
    # CSV laden
    print(f"input_file: {input_file}")
    df = pd.read_csv(input_file)

    # Gruppen definieren:
    # Alle Zeilen mit gleichem subject_ID, x und y gehören zusammen
    df["group_id"] = (
        df["subject_ID"].astype(str) + "_" +
        df["x"].astype(str) + "_" +
        df["y"].astype(str)
    )

    # Gruppenweiser Split
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(df, groups=df["group_id"])
    )

    rand_train_df = df.iloc[train_idx].drop(columns=["group_id"])
    rand_test_df = df.iloc[test_idx].drop(columns=["group_id"])


    mapping = {
        "labels.csv": (
            "./splits/random_train.csv",
            "./splits/random_test.csv"
        ),
        "norm_labels.csv": (
            "./splits/norm_random_train.csv",
            "./splits/norm_random_test.csv"
        )
    }

    filename = os.path.basename(input_file)
    print(f"\nfilename: {filename}")

    try:
        train_name, test_name = mapping[filename]
    except KeyError:
        raise ValueError(f"Unknown input file: {input_file}")

    # Speichern
    rand_train_df.to_csv(train_name, index=False)
    print(f"The file '{os.path.basename(train_name)}' is created! --> Train samples: {len(rand_train_df)}")

    rand_test_df.to_csv(test_name, index=False)
    print(f"The file '{os.path.basename(test_name)}' is created! --> Test samples: {len(rand_test_df)}\n")



def subjects_split(input_file):

    df = pd.read_csv(input_file)

    subjects = df["subject_ID"].unique()

    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    sub_train_df = df[df["subject_ID"].isin(train_subjects)]
    sub_test_df = df[df["subject_ID"].isin(test_subjects)]


    mapping = {
        "norm_labels.csv": (
            "./splits/norm_subject_train.csv",
            "./splits/norm_subject_test.csv"
        ),
        "labels.csv": (
            "./splits/subject_train.csv",
            "./splits/subject_test.csv"
        )
    }

    filename = os.path.basename(input_file)
    print(f"\nfilename: {filename}")

    try:
        train_name, test_name = mapping[filename]
    except KeyError:
        raise ValueError(f"Unknown input file: {input_file}")


    sub_train_df.to_csv(train_name,index=False)
    print(f"The file '{os.path.basename(train_name)}' is created! --> Train samples: {len(sub_train_df)}")

    sub_test_df.to_csv(test_name,index=False)
    print(f"The file '{os.path.basename(test_name)}' is created! --> Test samples: {len(sub_test_df)}\n")

## ---------------------------------------------------------------------------------------------
## ------------------ Test Split: --------------------------------------------------------------
## ---------------------------------------------------------------------------------------------
def check_split(input_train, input_test):
    train_df = pd.read_csv(input_train)
    test_df = pd.read_csv(input_test)

    # Extract subject sets:
    train_subjects = set(train_df["subject_ID"].unique())
    test_subjects = set(test_df["subject_ID"].unique())

    # 1. check intersection (important)
    overlap = train_subjects.intersection(test_subjects)

    if len(overlap) == 0:
        print("\n************\nOk, No overlapping subjects between train and test!\n************\n")

    else:
        print("\n************\nERROR: Overlapping subjects detected!\n************\n")
        print(f"\nNumber of Overlapping subjects: {len(overlap)}\n")
        print(f"\n\nExample overlap:\n", list(overlap)[:10])

    # 2: full coverage sanity
    all_subjects_original = train_subjects.union(test_subjects)
    print(f"Total unique subjects in split: {len(all_subjects_original)}")

    # 3: Ensure no subjects appears in both files at all:
    train_counts = train_df["subject_ID"].value_counts()
    test_counts = test_df["subject_ID"].value_counts()

    common_rows = set(train_counts.index).intersection(set(test_counts.index))

    if common_rows:
        print("\n************\nWARNING: These subjects appeare in both splits:")
        print(list(common_rows)[:10])

    else:
        print("\n************\nOK, No subject appears in both splits at row level!!\n************\n")


# if files don't exist, download them from kaggle
def main():

    # Checking for Dataset:
    dataset_path = './dataset/images/'
    if not os.path.exists('./dataset/images/00002/00002/frames/00000.jpg'):
        print("\nI have to dowanload Dataset!")
        confirm = input("\nDo I download the Dataset? (y/n)")
        if confirm == 'y':
            download_kaggle()
    else:
        print(f"Dataset found: {dataset_path}")

    # Checking for file (labels.csv):
    if not os.path.exists('./dataset/labels.csv'):
        print(f"The file: 'labels.csv' is necessary, but doesn't find!\n Try to create it ...  ")
        reformat(dataset_path, "./dataset/")
        if os.path.exists('./dataset/labels.csv'):
            print(f"Now can find the file: 'labels.csv' !\n ")

    # Checking for normalize-file (norm_labels.csv):
    if not os.path.exists('./dataset/norm_labels.csv'):
        submit = input(
            f"The file: 'norm_labels.csv' is maybe needed, but doesn't find!\n Do I have to create it? (y/n)")
        if submit.lower() == 'y':
            normalize(dataset_path, "./dataset/")
            if os.path.exists('./dataset/norm_labels.csv'):
                print(f"Now can find the file: 'norm_labels.csv' !\n ")

    while(True):
        confirm_split = input("Create split-files for labels or norm_labels? (l/n):")
        if confirm_split.lower() == 'l':
            csv_path = './dataset/labels.csv'
            break
        elif confirm_split.lower() == 'n':
            csv_path = './dataset/norm_labels.csv'
            break
        else:
            raise ValueError("Invalid input. Please enter 'l' or 'n'.")

    # Checking for random-split-files:
    if not os.path.exists('./splits/random_train.csv') or not os.path.exists('./splits/random_test.csv'):
        # random_split('./dataset/labels.csv')
        random_split(csv_path)

    # Checking for subjects-split-files:
    if not os.path.exists('./splits/subject_train.csv') or not os.path.exists('./splits/subject_test.csv'):
        # subjects_split('./dataset/labels.csv')
        subjects_split(csv_path)
        split_test = input("\n Do I have to test SPLIT?(y/n)")
        print("\n")
        if split_test == 'y':
            if confirm_split.lower() == 'n':
                train_path, test_path = "./splits/norm_subject_train.csv", "./splits/norm_subject_test.csv"

            elif confirm_split.lower() == 'l':
                train_path, test_path = "./splits/subject_train.csv", "./splits/subject_test.csv"

            check_split(train_path, test_path)


if __name__ == "__main__":
    main()

