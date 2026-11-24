
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_excel("Dataset.xlsx")

print("Dataset Loaded Successfully!")

# 2. Remove unused columns
df = df.drop(["file_name", "duplicate_group"], axis=1)

# 3. Encode categorical columns
categorical_columns = [
    "file_type",
    "folder_type",
    "shared_status",
    "owner_type"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

# 4. Encode target label
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# 5. Separate features and target
X = df.drop("label", axis=1)
y = df["label"]

feature_columns = X.columns.tolist()

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# 9. Save model and encoders
model_bundle = {
    "model": model,
    "encoders": encoders,
    "label_encoder": label_encoder,
    "feature_columns": feature_columns
}

joblib.dump(model_bundle, "file_classification_bundle.pkl")

print("\nModel and encoders saved successfully!")
print("File: file_classification_bundle.pkl")

print("\nLabel Mapping:")

for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)
