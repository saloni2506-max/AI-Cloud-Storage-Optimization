
import pandas as pd

df = pd.read_excel(
    "Dataset_updated_file_types.xlsx",
    sheet_name="Dataset"
)

print("\n1. RECOMMENDATION BY FILE TYPE")
print(pd.crosstab(df["file_type"], df["label"]))

print("\n2. RECOMMENDATION BY IMPORTANCE")
print(pd.crosstab(df["is_important"], df["label"]))

print("\n3. RECOMMENDATION BY STARRED STATUS")
print(pd.crosstab(df["is_starred"], df["label"]))

print("\n4. AVERAGE FILE AGE BY RECOMMENDATION")
print(df.groupby("label")["file_age_days"].mean().round(2))

print("\n5. AVERAGE FILE SIZE BY RECOMMENDATION")
print(df.groupby("label")["file_size_mb"].mean().round(2))
