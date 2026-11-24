
import pandas as pd

df = pd.read_excel("Dataset.xlsx")

categorical_columns = [
    "file_type",
    "folder_type",
    "shared_status",
    "owner_type"
]

print("\n========== CATEGORY MAPPING ==========")

for column in categorical_columns:
    print(f"\n{column.upper()}")
    print(df[column].value_counts(dropna=False).to_string())

print("\n========== IMPORTANT FILE DISTRIBUTION ==========")

print(df["is_important"].value_counts())

print("\n========== STARRED FILE DISTRIBUTION ==========")

print(df["is_starred"].value_counts())
