import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "water_leakage.csv")
MODEL_PATH = os.path.join(BASE_DIR, "smartleak_xgb.pkl")
TARGET = "Leakage_Flag"

NUMERIC_FEATURES = [
    "Pressure",
    "Flow_Rate",
    "Temperature",
    "Vibration",
    "RPM",
    "Operational_Hours",
    "Latitude",
    "Longitude"
]

CATEGORICAL_FEATURES = [
    "Zone",
    "Block",
    "Pipe"
]

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Basic sanity checks
df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])

# -----------------------------
# TIME-BASED SPLIT (REALISTIC)
# -----------------------------
print("Performing time-based split...")

df = df.sort_values("Operational_Hours")

train_df = df.iloc[:int(0.7 * len(df))]
temp_df = df.iloc[int(0.7 * len(df)):]

# -----------------------------
# STRATIFIED VAL / TEST SPLIT
# -----------------------------
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df[TARGET],
    random_state=42
)

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df[TARGET]

X_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_val = val_df[TARGET]

X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_test = test_df[TARGET]

# -----------------------------
# PREPROCESSING
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
    ]
)

# -----------------------------
# MODEL
# -----------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# -----------------------------
# TRAIN
# -----------------------------
print("Training XGBoost model...")
pipeline.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
print("\nEvaluating on test set...")

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(pipeline, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
