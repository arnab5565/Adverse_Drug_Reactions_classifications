"""
ADR Model Testing - Quick Reference Cheat Sheet
"""

# ═══════════════════════════════════════════════════════════════════════════
# 1. SIMPLEST WAY - Use Pre-Built Script
# ═══════════════════════════════════════════════════════════════════════════

# Just run this in terminal:
# >>> python test_single_prediction.py

# This trains models and tests 3 examples + batch predictions


# ═══════════════════════════════════════════════════════════════════════════
# 2. TEST SINGLE PATIENT - Minimal Code
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd

# Create patient record (as DataFrame)
patient = pd.DataFrame({
    "age_group": ["41-60"],
    "gender": ["Female"],
    "dosage_form": ["Injection"],
    "drug_class": ["Anticoagulant"],
    "country": ["US"],
    "num_drugs": [3],
    "prior_adr": [1],
    "renal_impairment": [0],
    "hepatic_impairment": [1]
})

# Preprocess (apply label encoding)
from test_single_prediction import preprocess_test_data, le_dict
X_test = preprocess_test_data(patient, le_dict)

# Predict
from test_single_prediction import dt_pipe, lr_pipe, SEVERITY_LABELS
dt_pred = dt_pipe.predict(X_test)[0]
lr_pred = lr_pipe.predict(X_test)[0]

print(f"Decision Tree: {SEVERITY_LABELS[dt_pred]}")
print(f"Logistic Regression: {SEVERITY_LABELS[lr_pred]}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TEST FROM CSV FILE
# ═══════════════════════════════════════════════════════════════════════════

# Create patients.csv:
# age_group,gender,dosage_form,drug_class,country,num_drugs,prior_adr,renal_impairment,hepatic_impairment
# 41-60,Female,Injection,Anticoagulant,US,3,1,0,1
# 21-40,Male,Tablet,Antibiotic,CA,1,0,0,0

import pandas as pd
from test_single_prediction import preprocess_test_data, dt_pipe, SEVERITY_LABELS, le_dict

test_df = pd.read_csv("patients.csv")
X_test = preprocess_test_data(test_df, le_dict)
predictions = dt_pipe.predict(X_test)

test_df["severity_prediction"] = [SEVERITY_LABELS[p] for p in predictions]
print(test_df)


# ═══════════════════════════════════════════════════════════════════════════
# 4. VALID INPUT VALUES
# ═══════════════════════════════════════════════════════════════════════════

VALID_VALUES = {
    "age_group": ["0-20", "21-40", "41-60", "61-80", "80+"],
    "gender": ["Female", "Male"],
    "dosage_form": ["Tablet", "Injection", "Capsule", "Solution", "Patch", "Inhaler", "Unknown"],
    "drug_class": ["Antibiotic", "Anticoagulant", "NSAID", "Antihypertensive", 
                   "Chemotherapy", "Antidiabetic", "Antidepressant", "Other", "Statin"],
    "country": ["US", "CA", "UK", "DE", "Other"],
    "num_drugs": "1-8 (integer)",
    "prior_adr": "0 or 1",
    "renal_impairment": "0 or 1",
    "hepatic_impairment": "0 or 1"
}

# OUTPUT CLASSES
OUTPUT_CLASSES = {
    0: "No Reaction",
    1: "Hospitalization",
    2: "Life-Threatening / Disabling",
    3: "Death"
}


# ═══════════════════════════════════════════════════════════════════════════
# 5. BATCH PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from test_single_prediction import preprocess_test_data, dt_pipe, lr_pipe, SEVERITY_LABELS, le_dict

# Create batch (multiple patients)
batch = pd.DataFrame({
    "age_group": ["41-60", "21-40", "80+", "61-80"],
    "gender": ["F", "M", "F", "M"],
    "dosage_form": ["Injection", "Tablet", "Patch", "Solution"],
    "drug_class": ["Anticoagulant", "Antibiotic", "NSAID", "Chemotherapy"],
    "country": ["US", "CA", "UK", "US"],
    "num_drugs": [3, 1, 2, 5],
    "prior_adr": [1, 0, 0, 1],
    "renal_impairment": [0, 0, 1, 1],
    "hepatic_impairment": [1, 0, 0, 1]
})

# Predict
X_batch = preprocess_test_data(batch, le_dict)
dt_preds = dt_pipe.predict(X_batch)
lr_preds = lr_pipe.predict(X_batch)

# Results
results = pd.DataFrame({
    "Patient_ID": [f"P{i}" for i in range(len(batch))],
    "Age": batch["age_group"],
    "DT_Prediction": [SEVERITY_LABELS[p] for p in dt_preds],
    "LR_Prediction": [SEVERITY_LABELS[p] for p in lr_preds]
})
print(results)


# ═══════════════════════════════════════════════════════════════════════════
# 6. WITH CONFIDENCE SCORES
# ═══════════════════════════════════════════════════════════════════════════

from test_single_prediction import get_prediction_confidence, preprocess_test_data

X_test = preprocess_test_data(patient, le_dict)

# Get prediction + confidence
dt_pred, dt_conf = get_prediction_confidence(dt_pipe, X_test)
lr_pred, lr_conf = get_prediction_confidence(lr_pipe, X_test)

print(f"Decision Tree: {SEVERITY_LABELS[dt_pred]} ({dt_conf:.1%} confidence)")
print(f"Logistic Regression: {SEVERITY_LABELS[lr_pred]} ({lr_conf:.1%} confidence)")


# ═══════════════════════════════════════════════════════════════════════════
# 7. INTERACTIVE INPUT
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd
from test_single_prediction import create_test_record, preprocess_test_data, dt_pipe, SEVERITY_LABELS, le_dict

# Get user input
age_group = input("Age [0-20/21-40/41-60/61-80/80+]: ")
gender = input("Gender [Female/Male]: ")
dosage_form = input("Drug form [Tablet/Injection/Capsule/Solution/Patch/Inhaler]: ")
drug_class = input("Drug class [Antibiotic/Anticoagulant/NSAID/...]: ")
country = input("Country [US/CA/UK/DE/Other]: ")
num_drugs = int(input("Concurrent drugs [1-8]: "))
prior_adr = int(input("Prior ADR [0/1]: "))
renal_imp = int(input("Renal impairment [0/1]: "))
hepatic_imp = int(input("Hepatic impairment [0/1]: "))

# Create record
patient = create_test_record(age_group, gender, dosage_form, drug_class, country,
                             num_drugs, prior_adr, renal_imp, hepatic_imp)

# Predict
X_test = preprocess_test_data(patient, le_dict)
pred = dt_pipe.predict(X_test)[0]
conf = dt_pipe.predict_proba(X_test).max()

print(f"\n✓ Severity: {SEVERITY_LABELS[pred]}")
print(f"✓ Confidence: {conf:.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════

# Problem: Feature names don't match
# Solution: Use preprocess_test_data() which handles encoding automatically

# Problem: Unknown category in drug_class
# Solution: Always use values from VALID_VALUES, use "Other" as default

# Problem: "Shape mismatch" error
# Solution: Ensure all 9 features present: 
# [age_group, gender, dosage_form, drug_class, country, num_drugs, prior_adr, 
#  renal_impairment, hepatic_impairment]

# Problem: Confidence is None
# Solution: Some models don't support predict_proba. Use:
# try:
#     confidence = model.predict_proba(X_test).max()
# except AttributeError:
#     confidence = None


# ═══════════════════════════════════════════════════════════════════════════
# 9. COMPARING MODELS
# ═══════════════════════════════════════════════════════════════════════════

# Decision Tree:
#   - Faster
#   - More interpretable
#   - May overfit

# Logistic Regression:
#   - More robust
#   - Better probabilities
#   - Slower

# Use both and average if predictions differ

from test_single_prediction import preprocess_test_data, dt_pipe, lr_pipe

X_test = preprocess_test_data(patient, le_dict)
dt_pred = dt_pipe.predict(X_test)[0]
lr_pred = lr_pipe.predict(X_test)[0]

if dt_pred == lr_pred:
    print(f"✓ CONSENSUS: Both models agree on {SEVERITY_LABELS[dt_pred]}")
else:
    print(f"⚠ DISAGREEMENT: DT={SEVERITY_LABELS[dt_pred]}, LR={SEVERITY_LABELS[lr_pred]}")
    # Use confidence scores to decide


# ═══════════════════════════════════════════════════════════════════════════
# 10. REAL-WORLD EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

# Scenario: 75-year-old female on anticoagulant injection with renal issues
# High-risk profile that should predict hospitalization or worse

high_risk_patient = pd.DataFrame({
    "age_group": ["61-80"],  # Older
    "gender": ["Female"],
    "dosage_form": ["Injection"],  # IV route (higher risk)
    "drug_class": ["Anticoagulant"],  # Blood thinner (interaction risk)
    "country": ["US"],
    "num_drugs": [4],  # Polypharmacy
    "prior_adr": [1],  # Has ADR history
    "renal_impairment": [1],  # Kidney issues (accumulation risk)
    "hepatic_impairment": [0]
})

X_test = preprocess_test_data(high_risk_patient, le_dict)
dt_pred = dt_pipe.predict(X_test)[0]
print(f"Expected: Hospitalization (class 1) or higher")
print(f"Actual: {SEVERITY_LABELS[dt_pred]}")
