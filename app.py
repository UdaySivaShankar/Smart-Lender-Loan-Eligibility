"""
app.py
------
Flask backend for the Smart Lender loan eligibility prediction system.

Routes:
    GET  /            -> home page
    GET  /predict      -> loan application input form
    POST /submit       -> processes form, runs prediction, renders result
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "rdf.pkl")

app = Flask(__name__)

# ----------------------------------------------------------------------
# Load trained model bundle once at startup
# ----------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
label_encoders = bundle["label_encoders"]
scaler = bundle["scaler"]
scale_cols = bundle["scale_cols"]
feature_columns = bundle["feature_columns"]
model_name = bundle.get("model_name", "XGBoost")
train_accuracy = bundle.get("train_accuracy", 0.0)
test_accuracy = bundle.get("test_accuracy", 0.0)


def build_feature_vector(form):
    """Convert raw form input into the encoded/scaled feature vector
    the model expects (same preprocessing as training)."""

    raw = {
        "Gender": form.get("gender"),
        "Married": form.get("married"),
        "Dependents": form.get("dependents"),
        "Education": form.get("education"),
        "Self_Employed": form.get("self_employed"),
        "ApplicantIncome": float(form.get("applicant_income", 0)),
        "CoapplicantIncome": float(form.get("coapplicant_income", 0)),
        "LoanAmount": float(form.get("loan_amount", 0)),
        "Loan_Amount_Term": float(form.get("loan_amount_term", 360)),
        "Credit_History": float(form.get("credit_history", 1)),
        "Property_Area": form.get("property_area"),
    }

    encoded = {}
    for col, value in raw.items():
        if col in label_encoders:
            le = label_encoders[col]
            # Guard against unseen categories
            value = value if value in le.classes_ else le.classes_[0]
            encoded[col] = le.transform([value])[0]
        else:
            encoded[col] = value

    row_df = pd.DataFrame([[encoded[col] for col in feature_columns]], columns=feature_columns)
    row_df[scale_cols] = scaler.transform(row_df[scale_cols])

    return row_df, raw


@app.route("/")
def home():
    return render_template("home.html", model_name=model_name,
                            train_accuracy=round(train_accuracy * 100, 1),
                            test_accuracy=round(test_accuracy * 100, 1))


@app.route("/predict")
def predict_form():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        feature_vector, raw_inputs = build_feature_vector(request.form)
        pred = model.predict(feature_vector)[0]
        proba = model.predict_proba(feature_vector)[0]

        status_encoder = label_encoders["Loan_Status"]
        label = status_encoder.inverse_transform([pred])[0]  # 'Y' or 'N'
        approved = (label == "Y")
        confidence = round(float(np.max(proba)) * 100, 1)

        return render_template(
            "submit.html",
            approved=approved,
            confidence=confidence,
            inputs=raw_inputs,
            model_name=model_name,
        )
    except Exception as exc:  # pragma: no cover
        return render_template("submit.html", error=str(exc))


if __name__ == "__main__":
    # host=0.0.0.0 so it is reachable when deployed (e.g. IBM Cloud)
    app.run(host="0.0.0.0", port=5000, debug=True)
