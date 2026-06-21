
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

            #print(f"Processing subject {subject_id}")

            # total_subjects += 1

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
                #skipped_subjects += 1
                continue

            if not dotinfo_file.exists():
                print(f"Skipping {subject_id}: dotInfo.json missing")
                #skipped_subjects += 1
                continue

            if not frames_folder.exists():
                print(f"Skipping {subject_id}: frames folder missing")
                #skipped_subjects += 1
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

    # Speichern
    rand_train_df.to_csv("./splits/random_train.csv", index=False)
    print(f"\n   The file 'random_train.csv' is created!    -->     Train samples: {len(rand_train_df)}")

    rand_test_df.to_csv("./splits/random_test.csv", index=False)
    print(f"\n   The file 'random_test.csv' is created!     -->     Test samples: {len(rand_test_df)}")

    # print(f"Train samples: {len(train_df)}")
    # print(f"Test samples: {len(test_df)}")



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

    sub_train_df.to_csv("./splits/subject_train.csv",index=False)
    print(f"\n   The file: 'subject_train.csv' is created!   -->     Train samples: {len(sub_train_df)}")

    sub_test_df.to_csv("./splits/subject_test.csv",index=False)
    print(f"\n   The file: 'subject_test.csv' is created!    -->     Train samples: {len(sub_test_df)}")

## ---------------------------------------------------------------------------------------------
## ------------------ Test Split: --------------------------------------------------------------
## ---------------------------------------------------------------------------------------------
def test_split(input_train, input_test):
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

## ---------------------------------------------------------------------------------------------



#if files don't exist, download them from kaggle
def main():

    dataset_path = './dataset/images'
    # if not os.path.exists('./dataset/gaze-capture-20gb/00002/00002/frames/00000.jpg'):
    if not os.path.exists('./dataset/images/00002/00002/frames/00000.jpg'):
        print("\nI have to dowanload Dataset!")
        confirm = input("\nDo I download the Dataset? (y/n)")
        if confirm == 'y':
            download_kaggle()
    else:
        print(f"Dataset found: {dataset_path}")
    if not os.path.exists('./dataset/labels.csv'):
        print(f"The file: 'labels.csv' is necessary, but doesn't find!\n Try to create it ...  ")
        reformat(dataset_path, "./dataset/")
        if os.path.exists('./dataset/labels.csv'):
            print(f"Now can find the file: 'labels.csv' !\n ")

    if not os.path.exists('./splits/random_train.csv') or not os.path.exists('./splits/random_test.csv'):
        random_split('./dataset/labels.csv')

    if not os.path.exists('./splits/subject_train.csv') or not os.path.exists('./splits/subject_test.csv'):
        subjects_split('./dataset/labels.csv')
        split_test = input("\n Do I have to test SPLIT?(y/n)")
        print("\n")
        if split_test == 'y':
            test_split('./splits/subject_train.csv', './splits/subject_test.csv')





if __name__ == "__main__":
    main()

