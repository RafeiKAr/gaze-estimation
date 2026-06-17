
"""
# goals:
    a) random:
        random_train.csv
        random_test.csv

    b) subjectindependit
        subject_train.csv
        subject_test.csv

    c) For few-Shot-Personalisation --> subject_test.csv: (for example: subject: 50 --> 200 images)
        subject_test_calibration.scv  --> (10  images  Calibration)
        subject_test_evaluation.csv   --> (190 images  Evaluation)


"""
import pandas as pd
from sklearn.model_selection import train_test_split


# step 1: load csv-file
df = pd.read_csv("labels2.csv")


# step 2: run random-spilt
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    train_size=0.8
)

# step 3: save
train_df.to_csv(
    "random_train.csv",
    index=False
)
print("\nThe file: 'random_train.csv' is created!")

test_df.to_csv(
    "random_test.csv",
    index=False
)
print("\nThe file: 'random_test.csv' is created!")
