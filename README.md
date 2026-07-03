# Smart Lender — Loan Eligibility Prediction System

A machine-learning powered web application that predicts loan applicant
creditworthiness, built with **Python, Flask, scikit-learn and XGBoost**.

## Project Structure

```
smart-lender/
├── app.py                    # Flask backend (routes, prediction logic)
├── requirements.txt          # Python dependencies
├── Procfile                  # For gunicorn-based cloud deployment
├── dataset/
│   ├── generate_dataset.py   # Generates the synthetic loan_data.csv
│   └── loan_data.csv         # Training dataset
├── model/
│   ├── train_model.py        # Full ML pipeline (EDA, preprocessing, training)
│   └── rdf.pkl                # Saved best model bundle (pickled)
├── static/
│   ├── css/style.css         # App styling
│   └── plots/                # EDA & model comparison charts (PNG)
└── templates/
    ├── home.html              # Landing page
    ├── predict.html           # Loan application input form
    └── submit.html            # Prediction result page
```

## ⚠️ About the dataset

This project was built in a sandboxed environment without general internet
access, so the original Google Sheets loan dataset could not be downloaded.
`dataset/generate_dataset.py` generates a **synthetic dataset (614 rows)**
with the exact same schema, realistic feature correlations, and a similar
missing-value pattern as the classic "Loan Prediction" dataset (Gender,
Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
Property_Area, Loan_Status).

**To use your real dataset:** replace `dataset/loan_data.csv` with your own
file (same column names), then re-run `python model/train_model.py` to
retrain and regenerate `model/rdf.pkl`. Model accuracy will change to reflect
the real data — the 96.5% / 76.4% train/test figures currently baked into
`rdf.pkl` are from the synthetic data, not the original 94.7% / 81.1%
figures quoted in the project brief.

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the dataset
python dataset/generate_dataset.py

# 4. Train the models and produce model/rdf.pkl
python model/train_model.py

# 5. Run the Flask app
python app.py
```

Then open **https://smart-lender-loan-eligibility.onrender.com/** in your browser.

## Pipeline Overview (`model/train_model.py`)

1. **Dataset Collection** — loads `dataset/loan_data.csv`
2. **EDA** — count plots, distribution plots, and bar charts saved to `static/plots/`
3. **Preprocessing** — missing values filled (mean for numeric, mode for
   categorical), categorical variables label-encoded, numeric features scaled
4. **Train/Test Split** — 80/20 stratified split
5. **Model Training** — Decision Tree, Random Forest, KNN, XGBoost
6. **Evaluation** — accuracy, confusion matrix, classification report, 5-fold CV
7. **Model Selection & Storage** — XGBoost is saved to `model/rdf.pkl` along
   with its label encoders and scaler, ready for the Flask app to load

## Retraining with new data

Any time you update `dataset/loan_data.csv`, just re-run:

```bash
python model/train_model.py
```

This regenerates `model/rdf.pkl` and all EDA/comparison charts in
`static/plots/`.
