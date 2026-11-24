
import joblib

# Load trained model
bundle = joblib.load("file_classification_bundle.pkl")

print("Model loaded successfully!")

print("\nExpected input features:")
print(bundle["feature_columns"])

print("\nSupported file types:")
print(bundle["encoders"]["file_type"].classes_)

print("\nSupported folder categories:")
print(bundle["encoders"]["folder_type"].classes_)

print("\nPrediction labels:")
print(bundle["label_encoder"].classes_)
