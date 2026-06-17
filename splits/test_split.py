
# check the Spliting

import pandas as pd

# load splits:
train_df = pd.read_csv("subject_train.csv")
test_df  = pd.read_csv("subject_test.csv")

# Extract subject sets:
train_subjects = set(train_df["subject_ID"].unique())
test_subjects  = set(test_df["subject_ID"].unique())

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


"""
Output:

************
Ok, No overlapping subjects between train and test!
************

Total unique subjects in split: 262

************
OK, No subject appears in both splits at row level!!
************


"""