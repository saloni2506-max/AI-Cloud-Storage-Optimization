
import pandas as pd

df = pd.read_excel(
    "Dataset_updated_file_types.xlsx",
    sheet_name="Dataset"
)

print("\nFILE TYPE DISTRIBUTION:")
print(df["file_type"].value_counts())

print("\nRECOMMENDATION DISTRIBUTION:")
print(df["label"].value_counts())

print("\nIMPORTANT FILE DISTRIBUTION:")
print(df["is_important"].value_counts())

print("\nFOLDER CATEGORY DISTRIBUTION:")
print(df["folder_type"].value_counts())
