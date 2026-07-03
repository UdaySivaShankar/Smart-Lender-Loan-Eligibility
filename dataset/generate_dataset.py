"""
generate_dataset.py
--------------------
Generates a synthetic Loan Eligibility dataset (loan_data.csv) that mirrors
the structure of the classic Loan Prediction dataset used in the
Smart Lender project (Gender, Married, Dependents, Education,
Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area, Loan_Status).

NOTE: This sandbox does not have general internet access, so the original
Google-Sheets source dataset could not be downloaded. This script builds a
realistic, statistically-similar synthetic dataset (614 rows, same schema,
same rough class balance and missing-value pattern as the well known
Analytics Vidhya "Loan Prediction" dataset) so the rest of the pipeline
(EDA, preprocessing, training, Flask app) runs exactly as it would on the
real data. If you have access to the original dataset, simply replace
dataset/loan_data.csv with it -- the column names/schema already match.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 614  # same size as the original Loan Prediction dataset

genders = np.random.choice(["Male", "Female"], size=N, p=[0.82, 0.18])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.58, 0.17, 0.17, 0.08])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.38, 0.38, 0.24])
credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.84, 0.16])

applicant_income = np.random.gamma(shape=3.2, scale=1800, size=N).astype(int) + 1500
coapplicant_income = np.where(
    married == "Yes",
    np.random.gamma(shape=1.8, scale=900, size=N).astype(int),
    0,
)

loan_amount = (
    (applicant_income + coapplicant_income) / np.random.uniform(28, 42, size=N)
).astype(int)
loan_amount = np.clip(loan_amount, 9, 700)

loan_term_choices = [360, 180, 120, 84, 60, 300, 240, 36, 12]
loan_term_probs = [0.72, 0.06, 0.03, 0.02, 0.03, 0.03, 0.03, 0.05, 0.03]
loan_amount_term = np.random.choice(loan_term_choices, size=N, p=loan_term_probs).astype(float)

# ---- Simulate loan approval outcome with realistic dependencies ----
score = (
    2.6 * credit_history
    + 0.55 * (education == "Graduate")
    + 0.35 * (married == "Yes")
    + 0.30 * (property_area == "Semiurban")
    - 0.20 * (property_area == "Rural")
    + 0.0003 * (applicant_income + coapplicant_income)
    - 0.006 * loan_amount
    + np.random.normal(0, 1.1, size=N)
)
loan_status = np.where(score > np.percentile(score, 31.3), "Y", "N")  # ~68.7% approved, matches original ratio

df = pd.DataFrame({
    "Loan_ID": [f"LP{str(i).zfill(6)}" for i in range(1, N + 1)],
    "Gender": genders,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

# ---- Inject realistic missing values (mirrors original dataset's NA pattern) ----
def inject_na(col, frac):
    idx = np.random.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

inject_na("Gender", 0.021)
inject_na("Married", 0.008)
inject_na("Dependents", 0.025)
inject_na("Self_Employed", 0.052)
inject_na("LoanAmount", 0.036)
inject_na("Loan_Amount_Term", 0.023)
inject_na("Credit_History", 0.081)

df.to_csv("dataset/loan_data.csv", index=False)
print(f"Generated dataset/loan_data.csv with {len(df)} rows and {df.shape[1]} columns")
print(df.isna().sum())
print("\nLoan_Status distribution:")
print(df["Loan_Status"].value_counts(normalize=True))
