# --------------------------------------------------------------
# soals:
#       subject_train.scv
#       subject_test.csv
#---------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("labels2.csv")

subjects = df["subject_ID"].unique()

train_subjects, test_subjects = train_test_split(
    subjects,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_df = df[df["subject_ID"].isin(train_subjects)]
test_df = df[df["subject_ID"].isin(test_subjects)]

train_df.to_csv(
    "subject_train.csv",
    index=False
)
print("\nThe file: 'subject_train.csv' is created!")

test_df.to_csv(
    "subject_test.csv",
    index=False
)
print("\nThe file: 'subject_test.csv' is created!")


"""
# step 5_1: train_subjects.csv
pd.DataFrame(train_subjects, columns=["subject_ID"]).to_csv(
    "train_subjects.csv",
    index=False
)

print("\nThe file: 'train_subjects.csv' is created!")


# step 5_2: test_subjects.csv
pd.DataFrame(test_subjects, columns=["subject_ID"]).to_csv(
    "test_subjects.csv",
    index=False
)

print("\nThe file: 'test_subjects.csv' is created!")
"""













"""
# step 4: create train_subjects and test_subjects:
subjects = df["subject_ID"].unique()

train_subjects, test_subjects = train_test_split(
    subjects,
    test_size=0.2,
    random_state=42
)


# step 5: save the files
# step 5_1: train_subjects.csv
pd.DataFrame(train_subjects, columns=["subject_ID"]).to_csv(
    "train_subjects.csv",
    index=False
)

print("\nThe file: 'train_subjects.csv' is created!")


# step 5_2: test_subjects.csv
pd.DataFrame(test_subjects, columns=["subject_ID"]).to_csv(
    "test_subjects.csv",
    index=False
)

print("\nThe file: 'test_subjects.csv' is created!")
"""