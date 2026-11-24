
import pandas as pd

df = pd.read_csv("google_drive_files.csv")

print("TOTAL GOOGLE DRIVE FILES:", len(df))

print("\nFILE TYPES:")
print(df["file_type"].value_counts(dropna=False))

print("\nMISSING FILE TYPES:")
print(df["file_type"].isna().sum())

print("\nFIRST 15 FILES:")
print(df[["file_name", "file_type"]].head(15).to_string(index=False))
