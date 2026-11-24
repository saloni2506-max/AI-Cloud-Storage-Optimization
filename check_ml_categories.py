
import pandas as pd

df = pd.read_csv("drive_ml_dataset.csv")

print("\nFile type counts:")
print(df["file_type"].value_counts())

print("\nFolder category counts:")
print(df["folder_type"].value_counts())

print("\nImportance values:")
print(df["is_important"].value_counts(dropna=False))

print("\nRecords without importance:")
print(df[df["is_important"].isna()][
    ["file_name", "file_type", "folder_type"]
].head(15))
