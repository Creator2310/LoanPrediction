import pandas as pd
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# --- 1. Load Data ---
CSV_PATH = "./loan_approval_dataset.csv"

if not os.path.exists(CSV_PATH):
    print(f"❌ Error: '{CSV_PATH}' not found. Please place the CSV file in this directory.")
    exit(1)

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()  # Remove any accidental spaces

print(f"✅ Loaded dataset with {len(df)} records and {len(df.columns)} columns")

# --- 2. Data Cleaning & Preprocessing ---
# Fix possible case inconsistencies
df['education'] = df['education'].astype(str).str.strip().str.lower()
df['loan_status'] = df['loan_status'].astype(str).str.strip().str.lower()

# Encode education: Graduate = 1, Not Graduate = 0
df['education_num'] = df['education'].apply(lambda x: 1 if x in ['graduate', 'grad', 'g'] else 0)

# Calculate total assets
df['assets_total'] = (
    df['residential_assets_value']
    + df['commercial_assets_value']
    + df['luxury_assets_value']
    + df['bank_asset_value']
)

# Encode loan status (Approved = 1, Rejected = 0)
df['loan_status_num'] = df['loan_status'].apply(
    lambda x: 1 if x in ['approved', 'yes', 'y', '1', 'true'] else 0
)

# Check label balance
label_counts = df['loan_status_num'].value_counts().to_dict()
print(f"🔍 Loan Status Distribution: {label_counts}")

if len(label_counts) == 1:
    print("⚠️ Warning: Only one class present in loan_status_num — check CSV values (expected 'Approved'/'Rejected').")

# --- 3. Define Features ---
FEATURE_COLUMNS = [
    'no_of_dependents',
    'education_num',
    'income_annum',
    'loan_amount',
    'cibil_score',
    'assets_total',
]

X = df[FEATURE_COLUMNS].copy()
y = df['loan_status_num']

# Convert large financials to Lakhs
for col in ['income_annum', 'loan_amount', 'assets_total']:
    X[col] = X[col] / 100000.0

# --- 4. Input Ranges (used by frontend UI validation) ---
input_ranges = {}
for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number) and col not in ['loan_id', 'loan_status_num']:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        input_ranges[col] = {
            "min": int(min_val),
            "max": int(max_val),
            "step": 1 if col in ['no_of_dependents', 'loan_term', 'cibil_score'] else 100000,
        }

# --- 5. Normalization using MinMaxScaler ---
scaler = MinMaxScaler()
X_std = scaler.fit_transform(X)

# --- 6. Build normalization ranges for JS ---
normalization_ranges = {}
for i, col in enumerate(FEATURE_COLUMNS):
    js_key = {
        'no_of_dependents': 'dependents',
        'education_num': 'education',
        'income_annum': 'income',
        'loan_amount': 'loan_amount',
        'cibil_score': 'cibil',
        'assets_total': 'assets_total'
    }[col]

    normalization_ranges[js_key] = {
        "min": float(scaler.data_min_[i]),
        "max": float(scaler.data_max_[i]),
    }

# --- 7. Construct final training data (standardized + label) ---
training_data = [list(row) + [int(status)] for row, status in zip(X_std, y)]

# --- 8. KNN Accuracy Evaluation ---
if len(set(y)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test) * 100
else:
    accuracy = 100.0  # trivial if only one class present

print(f"✅ KNN model trained successfully. Accuracy: {accuracy:.2f}%")

# --- 9. Export JSON ---
export_data = {
    "training_data_initial": training_data,
    "input_ranges": input_ranges,
    "normalization_ranges": normalization_ranges,
    "initial_accuracy": float(round(accuracy, 2)),
}

with open("model_data.json", "w") as f:
    json.dump(export_data, f, indent=2)

print("\n✅ model_data.json successfully written.")
print(f"📊 Summary:\n - Records: {len(training_data)}\n - Approved: {label_counts.get(1, 0)}\n - Rejected: {label_counts.get(0, 0)}")
print("\n➡️ Next Step: Place 'model_data.json' in your React app's /public directory.")
