"""
train_model.py
---------------
End-to-end training pipeline for the Smart Lender loan eligibility
prediction system.

Steps:
 1. Load dataset
 2. Exploratory Data Analysis (count plots, distribution plots, bar charts)
 3. Data preprocessing (missing values, encoding, scaling)
 4. Train/test split
 5. Train Decision Tree, Random Forest, KNN, XGBoost
 6. Evaluate (accuracy, confusion matrix, classification report, CV)
 7. Select best model (XGBoost) and save as rdf.pkl with encoders/scaler
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
MODEL_PATH = os.path.join(BASE_DIR, "model", "rdf.pkl")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Dataset Collection
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ----------------------------------------------------------------------
sns.set_theme(style="whitegrid")

# Count plot: Loan_Status
plt.figure(figsize=(5, 4))
sns.countplot(x="Loan_Status", data=df, palette="viridis")
plt.title("Loan Approval Status Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "countplot_loan_status.png"))
plt.close()

# Count plot: Gender vs Loan_Status
plt.figure(figsize=(5, 4))
sns.countplot(x="Gender", hue="Loan_Status", data=df, palette="magma")
plt.title("Loan Status by Gender")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "countplot_gender_loan_status.png"))
plt.close()

# Distribution plot: ApplicantIncome
plt.figure(figsize=(5, 4))
sns.histplot(df["ApplicantIncome"].dropna(), kde=True, color="steelblue")
plt.title("Applicant Income Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "distplot_applicant_income.png"))
plt.close()

# Distribution plot: LoanAmount
plt.figure(figsize=(5, 4))
sns.histplot(df["LoanAmount"].dropna(), kde=True, color="seagreen")
plt.title("Loan Amount Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "distplot_loan_amount.png"))
plt.close()

# Bar chart: Approval rate by Property_Area
plt.figure(figsize=(5, 4))
approval_by_area = (
    df.assign(Approved=(df["Loan_Status"] == "Y").astype(int))
    .groupby("Property_Area")["Approved"]
    .mean()
)
approval_by_area.plot(kind="bar", color=["#4c72b0", "#55a868", "#c44e52"])
plt.title("Approval Rate by Property Area")
plt.ylabel("Approval Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "barchart_approval_by_area.png"))
plt.close()

# Bar chart: Approval rate by Credit_History
plt.figure(figsize=(5, 4))
approval_by_credit = (
    df.assign(Approved=(df["Loan_Status"] == "Y").astype(int))
    .groupby("Credit_History")["Approved"]
    .mean()
)
approval_by_credit.plot(kind="bar", color=["#c44e52", "#55a868"])
plt.title("Approval Rate by Credit History")
plt.ylabel("Approval Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "barchart_approval_by_credit.png"))
plt.close()

print(f"EDA plots saved to {PLOTS_DIR}")

# ----------------------------------------------------------------------
# 3. Data Preprocessing & Feature Engineering
# ----------------------------------------------------------------------
data = df.drop(columns=["Loan_ID"]).copy()

# Handle missing values -- categorical: mode, numerical: mean
categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
numerical_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

for col in numerical_cols:
    data[col] = data[col].fillna(data[col].mean())

# Encode categorical variables
label_encoders = {}
encode_cols = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area", "Loan_Status",
]
for col in encode_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature scaling for numerical columns
scale_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
scaler = StandardScaler()
data[scale_cols] = scaler.fit_transform(data[scale_cols])

FEATURE_COLUMNS = [c for c in data.columns if c != "Loan_Status"]

X = data[FEATURE_COLUMNS]
y = data["Loan_Status"]

# ----------------------------------------------------------------------
# 4. Train-Test Split
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 5. Model Training
# ----------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "XGBoost": XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    ),
}

results = {}

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)
    cm = confusion_matrix(y_test, test_pred)
    report = classification_report(y_test, test_pred, target_names=["Rejected", "Approved"])

    results[name] = {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "cv_mean": cv_scores.mean(),
        "confusion_matrix": cm,
    }

    print(f"\n--- {name} ---")
    print(f"Train Accuracy: {train_acc*100:.1f}%")
    print(f"Test Accuracy:  {test_acc*100:.1f}%")
    print(f"CV Mean Accuracy (5-fold): {cv_scores.mean()*100:.1f}%")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

# ----------------------------------------------------------------------
# 6. Model Comparison Chart
# ----------------------------------------------------------------------
plt.figure(figsize=(6, 4))
names = list(results.keys())
test_accs = [results[n]["test_acc"] * 100 for n in names]
train_accs = [results[n]["train_acc"] * 100 for n in names]

x_pos = np.arange(len(names))
width = 0.35
plt.bar(x_pos - width / 2, train_accs, width, label="Train Accuracy", color="#4c72b0")
plt.bar(x_pos + width / 2, test_accs, width, label="Test Accuracy", color="#dd8452")
plt.xticks(x_pos, names, rotation=15)
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison: Train vs Test Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"))
plt.close()

# ----------------------------------------------------------------------
# 7. Best Model Selection & Storage
# ----------------------------------------------------------------------
# XGBoost is selected as the production model per the Smart Lender project
# spec (best generalization/robustness trade-off observed across
# cross-validation runs on the real-world dataset). On this synthetic
# dataset the four models score closely together; swap the key below to
# `max(results, key=lambda n: results[n]["test_acc"])` if you want pure
# test-accuracy-based auto-selection instead.
best_name = "XGBoost"
best_model = results[best_name]["model"]

print("\n" + "=" * 60)
print(f"BEST MODEL SELECTED: {best_name}")
print(f"  Train Accuracy: {results[best_name]['train_acc']*100:.1f}%")
print(f"  Test Accuracy:  {results[best_name]['test_acc']*100:.1f}%")
print("=" * 60)

bundle = {
    "model": best_model,
    "model_name": best_name,
    "label_encoders": label_encoders,
    "scaler": scaler,
    "scale_cols": scale_cols,
    "feature_columns": FEATURE_COLUMNS,
    "train_accuracy": results[best_name]["train_acc"],
    "test_accuracy": results[best_name]["test_acc"],
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)

print(f"\nSaved best model bundle to {MODEL_PATH}")
