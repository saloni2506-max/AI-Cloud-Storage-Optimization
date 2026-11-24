import pandas as pd

df = pd.read_csv("drive_folder_mapped.csv")

print("Total rows:", len(df))
print("\nAvailable columns:")
print(df.columns.tolist())