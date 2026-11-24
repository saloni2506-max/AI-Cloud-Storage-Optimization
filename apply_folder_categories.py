
import pandas as pd
import ast

# -----------------------------------
# 1. Load Google Drive Files
# -----------------------------------

files_df = pd.read_csv("google_drive_files.csv", encoding="utf-8")

# Load manually classified folders
folders_df = pd.read_csv("folder_category_review.csv", encoding="cp1252")

# -----------------------------------
# 2. Prepare Folder Categories
# -----------------------------------

folder_categories = dict(
    zip(
        folders_df["id"].astype(str),
        folders_df["folder_category"].fillna("").str.strip()
    )
)

# -----------------------------------
# 3. Extract Parent Folder IDs
# -----------------------------------

def get_parent_ids(value):
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []

files_df["parent_folders"] = files_df["parent_folders"].apply(
    get_parent_ids
)

# -----------------------------------
# 4. Assign Folder Categories
# -----------------------------------

def get_folder_category(parent_ids):

    for folder_id in parent_ids:

        category = folder_categories.get(str(folder_id), "")

        if category in [
            "College",
            "Personal",
            "Entertainment",
            "Temporary",
            "Project",
            "Work"
        ]:
            return category

    return "Unclassified"


files_df["folder_type"] = files_df["parent_folders"].apply(
    get_folder_category
)

# -----------------------------------
# 5. Display Results
# -----------------------------------

print("\n========== FOLDER CATEGORY RESULTS ==========")

print("Total files:", len(files_df))

print("\nCategory Summary:")

print(files_df["folder_type"].value_counts())

# -----------------------------------
# 6. Save Updated Dataset
# -----------------------------------

files_df.to_csv(
    "drive_folder_mapped.csv",
    index=False
)

print("\nUpdated file saved successfully!")
print("File: drive_folder_mapped.csv")
