import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------
# 1. Load Processed Dataset
# -----------------------------------

df = pd.read_csv("processed_dataset.csv")

print("Dataset Shape:")
print(df.shape)


# -----------------------------------
# 2. Separate Features and Target
# -----------------------------------

X = df.drop("label", axis=1)
y = df["label"]


print("\nFeatures (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print(y.name)


# -----------------------------------
# 3. Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 4. Display Dataset Sizes
# -----------------------------------

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# -----------------------------------
# 5. Display Target Distribution
# -----------------------------------

print("\nTraining Labels:")
print(y_train.value_counts())

print("\nTesting Labels:")
print(y_test.value_counts())