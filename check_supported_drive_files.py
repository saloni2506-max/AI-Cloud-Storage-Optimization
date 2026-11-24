
import pandas as pd
import joblib

df = pd.read_csv("drive_ml_eligible.csv")
bundle = joblib.load("file_classification_bundle.pkl")

mime_mapping = {
    "application/pdf": "PDF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PPTX",
    "text/csv": "CSV",
    "text/plain": "TXT",
    "image/jpeg": "JPG",
    "image/png": "PNG",
    "video/mp4": "MP4",
    "application/zip": "ZIP"
}

df["ml_file_type"] = df["file_type"].map(mime_mapping)

supported_types = bundle["encoders"]["file_type"].classes_

supported = df[
    df["ml_file_type"].isin(supported_types)
].copy()

unsupported = df[
    ~df["ml_file_type"].isin(supported_types)
].copy()

print("Total eligible records:", len(df))
print("Supported records:", len(supported))
print("Unsupported records:", len(unsupported))

print("\nSupported file types:")
print(supported["ml_file_type"].value_counts())

supported.to_csv("drive_ml_supported.csv", index=False)
unsupported.to_csv("drive_ml_unsupported.csv", index=False)

print("\nFiles saved successfully!")
