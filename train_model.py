
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "gestures.csv"
MODEL_FILE = "gesture_model.pkl"

df = pd.read_csv(DATA_FILE).dropna()

if "label" not in df.columns:
    raise ValueError("Dataset must contain a label column.")

X = df.drop(columns=["label"])
y = df["label"]

if X.shape[1] != 63:
    raise ValueError(f"Expected 63 features, found {X.shape[1]}.")

if y.nunique() < 2:
    raise ValueError("Collect at least two gesture classes.")

print("Dataset shape:", df.shape)
print("\nSamples per class:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))

joblib.dump(
    {"model": model, "features": list(X.columns)},
    MODEL_FILE
)

print(f"\nSaved model to {MODEL_FILE}")
