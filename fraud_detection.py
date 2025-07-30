# ===============================
# FRAUD DETECTION FULL PIPELINE
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from lightgbm import LGBMClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("Fraud.csv")
print(f"Dataset loaded with shape: {df.shape}")

print("\nFraud Distribution:")
print(df['isFraud'].value_counts(normalize=True))

print("\nTransaction Types in Fraud:")
print(df[df['isFraud']==1]['type'].value_counts())

# -----------------------------
# STEP 2: REMOVE LEAKAGE FEATURES
# -----------------------------
# Drop unique IDs
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Drop system flag to avoid rule-based leakage
if 'isFlaggedFraud' in df.columns:
    df.drop(['isFlaggedFraud'], axis=1, inplace=True)

# -----------------------------
# STEP 3: FEATURE ENGINEERING
# -----------------------------
# Encode transaction type
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Target variable
y = df['isFraud']
X = df.drop('isFraud', axis=1)

print("\nFinal Features:")
print(X.columns.tolist())

# -----------------------------
# STEP 4: TEMPORAL TRAIN-TEST SPLIT
# -----------------------------
df_sorted = df.sort_values(by='step')
max_step = df_sorted['step'].max()
split_step = int(max_step * 0.7)

train_data = df_sorted[df_sorted['step'] <= split_step]
test_data  = df_sorted[df_sorted['step'] > split_step]

X_train = train_data.drop('isFraud', axis=1)
y_train = train_data['isFraud']
X_val   = test_data.drop('isFraud', axis=1)
y_val   = test_data['isFraud']

print(f"\nTrain size: {X_train.shape}, Validation size: {X_val.shape}")
print("Fraud in Train:", y_train.mean(), "Fraud in Validation:", y_val.mean())

# -----------------------------
# STEP 5: TRAIN LIGHTGBM
# -----------------------------
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)

print("\nTraining model...")
model.fit(X_train, y_train)
print("Model training complete.")

# -----------------------------
# STEP 6: EVALUATE MODEL
# -----------------------------
y_pred = model.predict(X_val)
y_pred_prob = model.predict_proba(X_val)[:, 1]

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

roc_auc = roc_auc_score(y_val, y_pred_prob)
pr_auc = average_precision_score(y_val, y_pred_prob)
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"PR AUC Score: {pr_auc:.4f}")

cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Optional plots (if running in notebook or interactive mode)
# import matplotlib.pyplot as plt
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title("Confusion Matrix")
# plt.show()

# -----------------------------
# STEP 7: FEATURE IMPORTANCE
# -----------------------------
importances = model.feature_importances_
features = X_train.columns
fi = pd.DataFrame({'Feature': features, 'Importance': importances})
fi = fi.sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(fi.head(10))

# -----------------------------
# STEP 8: SAVE FULL PIPELINE
# -----------------------------
pipeline = {
    "model": model,
    "features": X_train.columns.tolist()
}

joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
print("\n✅ Pipeline saved as fraud_detection_pipeline.pkl")

# -----------------------------
# STEP 9: HOW TO USE PIPELINE
# -----------------------------
"""
Example Usage in Another Script:

import joblib
import pandas as pd

pipeline = joblib.load("fraud_detection_pipeline.pkl")
model = pipeline["model"]
features = pipeline["features"]

# Prepare new_data DataFrame with the same columns as features
# new_data = pd.DataFrame([...], columns=features)

y_pred = model.predict(new_data)
y_prob = model.predict_proba(new_data)[:,1]
"""
