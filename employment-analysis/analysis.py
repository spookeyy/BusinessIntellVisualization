# ============================================================
# Crop Yield Prediction - Employment Decision Tree Classifier
# Dataset: IBM HR Analytics (Kaggle)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report
)

# ─────────────────────────────────────────────
# 1. DATA LOADING AND EXPLORATION
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading Dataset")
print("=" * 50)

raw = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(f"Raw dataset shape: {raw.shape}")

# Map IBM HR columns → assignment columns
edu_map = {1: "High School", 2: "High School", 3: "Bachelor's", 4: "Master's", 5: "PhD"}

df = pd.DataFrame({
    "age":                    raw["Age"],
    "education_level":        raw["Education"].map(edu_map),
    "years_of_experience":    raw["TotalWorkingYears"],
    "technical_test_score":   raw["PerformanceRating"] * 25,   # scale 1–4 → 25–100
    "interview_score":        raw["JobSatisfaction"] * 2.5,    # scale 1–4 → 2.5–10
    "previous_employment":    raw["NumCompaniesWorked"].apply(lambda x: "Yes" if x > 1 else "No"),
    "suitable_for_employment": raw["Attrition"].apply(lambda x: "Yes" if x == "No" else "No"),
})

print("\nFirst 5 rows:")
print(df.head())
print(f"\nDataset shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. BASIC EDA
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Exploratory Data Analysis")
print("=" * 50)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["suitable_for_employment"].value_counts())

print("\nBasic Statistics:")
print(df.describe())

# Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Feature Distributions", fontsize=14)

axes[0, 0].hist(df["age"], bins=15, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Age")

axes[0, 1].hist(df["years_of_experience"], bins=15, color="steelblue", edgecolor="white")
axes[0, 1].set_title("Years of Experience")

axes[0, 2].hist(df["technical_test_score"], bins=10, color="steelblue", edgecolor="white")
axes[0, 2].set_title("Technical Test Score")

axes[1, 0].hist(df["interview_score"], bins=10, color="steelblue", edgecolor="white")
axes[1, 0].set_title("Interview Score")

df["education_level"].value_counts().plot(kind="bar", ax=axes[1, 1], color="steelblue")
axes[1, 1].set_title("Education Level")
axes[1, 1].tick_params(axis="x", rotation=30)

df["suitable_for_employment"].value_counts().plot(kind="bar", ax=axes[1, 2], color=["steelblue", "salmon"])
axes[1, 2].set_title("Suitable for Employment")

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150)
plt.show()
print("Saved: eda_distributions.png")

# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Data Preprocessing")
print("=" * 50)

df_encoded = df.copy()

le = LabelEncoder()
for col in ["education_level", "previous_employment", "suitable_for_employment"]:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Features and target
X = df_encoded.drop("suitable_for_employment", axis=1)
y = df_encoded["suitable_for_employment"]

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# 4. MODEL BUILDING
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Training Decision Tree")
print("=" * 50)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully.")

# ─────────────────────────────────────────────
# 5. MODEL VISUALIZATION
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Decision Tree Visualization")
print("=" * 50)

plt.figure(figsize=(20, 8))
plot_tree(
    model,
    feature_names=X.columns.tolist(),
    class_names=["Not Suitable", "Suitable"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree – Employment Prediction", fontsize=14)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()
print("Saved: decision_tree.png")

# ─────────────────────────────────────────────
# 6. PREDICTIONS ON TEST SET
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Predictions")
print("=" * 50)

y_pred = model.predict(X_test)
print("Sample predictions (first 10):", y_pred[:10])

# ─────────────────────────────────────────────
# 7. HYPOTHETICAL CANDIDATE PROFILES
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 7: Hypothetical Candidate Profiles")
print("=" * 50)

# education_level encoding: Bachelor's=0, High School=1, Master's=2, PhD=3
# previous_employment: No=0, Yes=1
candidates = pd.DataFrame({
    "age":                 [28,  45,  23],
    "education_level":     [2,   3,   1],   # Master's, PhD, High School
    "years_of_experience": [5,   20,  1],
    "technical_test_score":[75,  100, 50],
    "interview_score":     [7.5, 10,  5.0],
    "previous_employment": [1,   1,   0],   # Yes, Yes, No
})

label_map = {1: "Suitable", 0: "Not Suitable"}
preds = model.predict(candidates)
profiles = ["Candidate A (experienced Master's)", "Candidate B (senior PhD)", "Candidate C (fresh grad)"]

for profile, pred in zip(profiles, preds):
    print(f"  {profile}: {label_map[pred]}")

# ─────────────────────────────────────────────
# 8. MODEL EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 8: Model Evaluation")
print("=" * 50)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {acc:.4f} ({acc*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Suitable", "Suitable"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Suitable", "Suitable"],
            yticklabels=["Not Suitable", "Suitable"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# ─────────────────────────────────────────────
# BONUS: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("BONUS: Feature Importance")
print("=" * 50)

importances = model.feature_importances_
feat_df = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feat_df)

plt.figure(figsize=(8, 5))
feat_df.plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("Feature Importance – Decision Tree")
plt.ylabel("Importance Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Saved: feature_importance.png")

print("\nDone.")
