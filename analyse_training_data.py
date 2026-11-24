
import pandas as pd

# Load original training dataset
df = pd.read_excel("Dataset_updated_file_types.xlsx", sheet_name="Dataset")

print("TOTAL TRAINING RECORDS:", len(df))

print("\n1. FILE TYPE DISTRIBUTION")
print(df["file_type"].value_counts())

print("\n2. RECOMMENDATION LABEL DISTRIBUTION")
print(df["label"].value_counts())

print("\n3. IMPORTANT FILE DISTRIBUTION")
print(df["is_important"].value_counts())

print("\n4. MISSING VALUES")
print(df.isnull().sum())

print("\n5. DATASET INFORMATION")
print(df.info())
