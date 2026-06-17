
from kaggle.api.kaggle_api_extended import KaggleApi

import json
import csv
import os
from pathlib import Path


def extract_zip(zip_path, extract_to):
    # shutil.unpack_archive(zip_path, extract_to)

    DATASET_ROOT = Path(extract_to)

    csv_path = DATASET_ROOT / "labels.csv"

    with open(csv_path, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "image_name",
            "x",
            "y",
            "subject"
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

            total_subjects += 1

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

            with open(frames_file, "r") as f:
                frame_names = json.load(f)

            with open(dotinfo_file, "r") as f:
                dot_info = json.load(f)

            x_values = dot_info["XPts"]
            y_values = dot_info["YPts"]

            if len(frame_names) != len(x_values):
                print(
                    f"ERROR in {subject_id}: "
                    f"{len(frame_names)} frames but "
                    f"{len(x_values)} labels"
                )
                continue

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
        path='.',
        unzip=True
    )

    print("Download completed!")


#if files don't exist, download them from kaggle
if not os.path.exists('gaze_capture_20gb/00002/00002/frames/00000.jpg'):
    print("\nI have to dowanload Dataset!")
    confirm = input("\nDo I download the Dataset? (y/n)")
    if confirm == 'y':
        download_kaggle()
if not os.path.exists('gaze_capture_20gb/labels.csv'):
    extract_zip("gaze-capture-20gb-zip.zip", "gaze_capture_20gb")

