"""
=============================================================================
Test Individual Predictions on ADR Project
How to evaluate new patient data using trained models
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Import data loading and preprocessing
import sys
sys.path.insert(0, '.')

try:
    from fetch_faers_data import fetch_real_faers_data
    HAS_FAERS_FETCHER = True
except ImportError:
    HAS_FAERS_FETCHER = False

# ═══════════════════════════════════════════════════════════════════════════
# 1. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

SEVERITY_LABELS = ["No Reaction", "Hospitalization", "Life-Threatening/Disabling", "Death"]
AGE_GROUPS = ["0-20", "21-40", "41-60", "61-80", "80+"]
GENDERS = ["Female", "Male"]
DOSAGE_FORMS = ["Capsule", "Inhaler", "Injection", "Patch", "Solution", "Tablet", "Unknown"]
DRUG_CLASSES = ["Antibiotic", "Antidepressant", "Antidiabetic", "Anticoagulant", "Antihypertensive", 
                 "Chemotherapy", "NSAID", "Other", "Statin"]
COUNTRIES = ["CA", "DE", "GB", "US", "Other"]

def create_test_record(age_group="41-60", gender="Female", dosage_form="Tablet", 
                       drug_class="NSAID", country="US", num_drugs=2, 
                       prior_adr=0, renal_impairment=0, hepatic_impairment=0):
    """
    Create a single test record in the required format
    
    Args:
        age_group: One of ["0-20", "21-40", "41-60", "61-80", "80+"]
        gender: "Female" or "Male"
        dosage_form: Drug form (Tablet, Injection, Capsule, Solution, Patch, Inhaler)
        drug_class: Drug category (Antibiotic, NSAID, Anticoagulant, etc.)
        country: Country code (US, CA, UK, DE, Other)
        num_drugs: Number of concurrent medications (1-8)
        prior_adr: Prior adverse reaction (0 or 1)
        renal_impairment: Kidney impairment (0 or 1)
        hepatic_impairment: Liver impairment (0 or 1)
    
    Returns:
        DataFrame with single record
    """
    return pd.DataFrame({
        "age_group": [age_group],
        "gender": [gender],
        "dosage_form": [dosage_form],
        "drug_class": [drug_class],
        "country": [country],
        "num_drugs": [num_drugs],
        "prior_adr": [prior_adr],
        "renal_impairment": [renal_impairment],
        "hepatic_impairment": [hepatic_impairment]
    })

def preprocess_test_data(test_df, le_dict):
    """
    Apply the same preprocessing as training data
    
    Args:
        test_df: DataFrame with test records
        le_dict: Dictionary of fitted LabelEncoders for each categorical column
    
    Returns:
        Preprocessed feature matrix
    """
    test_df_enc = test_df.copy()
    
    # Apply label encoding using fitted encoders
    for col, le in le_dict.items():
        if col in test_df_enc.columns:
            try:
                test_df_enc[col] = le.transform(test_df_enc[col])
            except ValueError as e:
                print(f"  ⚠ Warning: Unknown value in {col}. Using first class as fallback.")
                test_df_enc[col] = le.transform([le.classes_[0]])[0]
    
    return test_df_enc.values

def get_prediction_confidence(model, X_test):
    """
    Get prediction with confidence scores
    
    Args:
        model: Trained model (should support predict_proba)
        X_test: Test data
    
    Returns:
        (prediction, confidence) tuple
    """
    prediction = model.predict(X_test)
    
    # Try to get probability estimates
    try:
        proba = model.predict_proba(X_test)
        confidence = np.max(proba, axis=1)[0]
    except AttributeError:
        # If model doesn't support predict_proba
        confidence = None
    
    return prediction[0], confidence

def print_prediction_report(patient_data, dt_pred, dt_conf, lr_pred, lr_conf):
    """Pretty print prediction results"""
    print("\n" + "="*70)
    print("  PREDICTION REPORT")
    print("="*70)
    
    print("\n📋 PATIENT DATA:")
    print("-" * 70)
    for col, val in patient_data.items():
        print(f"  {col:.<25} {val}")
    
    print("\n🎯 DECISION TREE PREDICTIONS:")
    print("-" * 70)
    print(f"  Severity Class ........... {SEVERITY_LABELS[dt_pred]} (class {dt_pred})")
    if dt_conf is not None:
        print(f"  Confidence ............... {dt_conf:.1%}")
    else:
        print(f"  Confidence ............... N/A")
    
    print("\n🎯 LOGISTIC REGRESSION PREDICTIONS:")
    print("-" * 70)
    print(f"  Severity Class ........... {SEVERITY_LABELS[lr_pred]} (class {lr_pred})")
    if lr_conf is not None:
        print(f"  Confidence ............... {lr_conf:.1%}")
    else:
        print(f"  Confidence ............... N/A")
    
    print("\n" + "="*70)

# ═══════════════════════════════════════════════════════════════════════════
# 2. TRAIN MODELS (minimal retraining)
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  ADR SEVERITY PREDICTOR - Test Individual Patient Records")
print("=" * 70)

# Load and preprocess data
print("\n[1] Loading training data...")

def generate_faers_data(n=5000):
    """Generate synthetic FAERS data for demo"""
    np.random.seed(42)
    age_bins = ["0-20", "21-40", "41-60", "61-80", "80+"]
    age_probs = [0.06, 0.18, 0.32, 0.30, 0.14]
    
    age = np.random.choice(age_bins, n, p=age_probs)
    gender = np.random.choice(["Female", "Male"], n, p=[0.56, 0.44])
    dosage = np.random.choice(["Tablet", "Injection", "Capsule", "Solution", "Patch", "Inhaler"], 
                              n, p=[0.35, 0.25, 0.18, 0.10, 0.07, 0.05])
    drug_cls = np.random.choice(["Antibiotic", "Anticoagulant", "NSAID", "Antihypertensive",
                                 "Chemotherapy", "Antidiabetic", "Antidepressant"], n)
    country = np.random.choice(["US", "CA", "UK", "DE", "Other"], n, p=[0.55, 0.27, 0.08, 0.05, 0.05])
    num_drugs = np.random.randint(1, 8, n)
    prior_adr = np.random.randint(0, 2, n)
    renal_imp = np.random.choice([0, 1], n, p=[0.75, 0.25])
    hepatic_imp = np.random.choice([0, 1], n, p=[0.80, 0.20])
    
    score = np.zeros(n)
    score += np.where(dosage == "Injection", 1.2, 0)
    score += np.where(dosage == "Tablet", 0.5, 0)
    score += np.where(drug_cls == "Chemotherapy", 1.5, 0)
    score += np.where(drug_cls == "Anticoagulant", 1.0, 0)
    score += np.where(drug_cls == "Antibiotic", 0.4, 0)
    score += np.where(age == "80+", 1.2, 0)
    score += np.where(age == "61-80", 0.7, 0)
    score += np.where(age == "0-20", 0.3, 0)
    score += num_drugs * 0.15
    score += prior_adr * 0.8
    score += renal_imp * 0.9
    score += hepatic_imp * 0.7
    score += np.random.normal(0, 0.5, n)
    
    thresholds = np.percentile(score, [50, 72, 88])
    severity = np.digitize(score, thresholds)
    
    return pd.DataFrame({
        "age_group": age, "gender": gender, "dosage_form": dosage,
        "drug_class": drug_cls, "country": country, "num_drugs": num_drugs,
        "prior_adr": prior_adr, "renal_impairment": renal_imp,
        "hepatic_impairment": hepatic_imp, "severity": severity
    })

df = generate_faers_data(2000)
print(f"  ✓ Loaded {df.shape[0]:,} training records")

# Preprocessing
print("\n[2] Preprocessing data...")
cat_cols = ["age_group", "gender", "dosage_form", "drug_class", "country"]
df_enc = df.copy()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df_enc.drop("severity", axis=1)
y = df_enc["severity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  ✓ Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# Train models
print("\n[3] Training models...")

dt_pipe = Pipeline([("clf", DecisionTreeClassifier(max_depth=8, random_state=42))])
dt_pipe.fit(X_train, y_train)
dt_acc = dt_pipe.score(X_test, y_test)
print(f"  ✓ Decision Tree trained (test acc: {dt_acc:.2%})")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
])
lr_pipe.fit(X_train, y_train)
lr_acc = lr_pipe.score(X_test, y_test)
print(f"  ✓ Logistic Regression trained (test acc: {lr_acc:.2%})")

# ═══════════════════════════════════════════════════════════════════════════
# 3. TEST ON SAMPLE RECORDS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  TESTING INDIVIDUAL PREDICTIONS")
print("="*70)

# Example 1: Low-risk patient (elderly with tablet NSAID)
print("\n\n[Example 1] Elderly patient - NSAID tablet (low to moderate risk)")
patient_1 = create_test_record(
    age_group="80+",
    gender="Female",
    dosage_form="Tablet",
    drug_class="NSAID",
    country="US",
    num_drugs=2,
    prior_adr=0,
    renal_impairment=1,
    hepatic_impairment=0
)
print("Input record:")
print(patient_1.to_string(index=False))

X_test_1 = preprocess_test_data(patient_1, le_dict)
dt_pred_1, dt_conf_1 = get_prediction_confidence(dt_pipe, X_test_1)
lr_pred_1, lr_conf_1 = get_prediction_confidence(lr_pipe, X_test_1)
print_prediction_report(patient_1.iloc[0].to_dict(), dt_pred_1, dt_conf_1, lr_pred_1, lr_conf_1)

# Example 2: High-risk patient (IV chemotherapy with multiple drugs)
print("\n\n[Example 2] High-risk patient - IV Chemotherapy with polypharmacy")
patient_2 = create_test_record(
    age_group="61-80",
    gender="Male",
    dosage_form="Injection",
    drug_class="Chemotherapy",
    country="US",
    num_drugs=5,
    prior_adr=1,
    renal_impairment=1,
    hepatic_impairment=1
)
print("Input record:")
print(patient_2.to_string(index=False))

X_test_2 = preprocess_test_data(patient_2, le_dict)
dt_pred_2, dt_conf_2 = get_prediction_confidence(dt_pipe, X_test_2)
lr_pred_2, lr_conf_2 = get_prediction_confidence(lr_pipe, X_test_2)
print_prediction_report(patient_2.iloc[0].to_dict(), dt_pred_2, dt_conf_2, lr_pred_2, lr_conf_2)

# Example 3: Young patient - Antibiotic capsule (minimal risk)
print("\n\n[Example 3] Young patient - Antibiotic capsule (minimal risk)")
patient_3 = create_test_record(
    age_group="21-40",
    gender="Female",
    dosage_form="Capsule",
    drug_class="Antibiotic",
    country="US",
    num_drugs=1,
    prior_adr=0,
    renal_impairment=0,
    hepatic_impairment=0
)
print("Input record:")
print(patient_3.to_string(index=False))

X_test_3 = preprocess_test_data(patient_3, le_dict)
dt_pred_3, dt_conf_3 = get_prediction_confidence(dt_pipe, X_test_3)
lr_pred_3, lr_conf_3 = get_prediction_confidence(lr_pipe, X_test_3)
print_prediction_report(patient_3.iloc[0].to_dict(), dt_pred_3, dt_conf_3, lr_pred_3, lr_conf_3)

# ═══════════════════════════════════════════════════════════════════════════
# 4. BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*70)
print("  BATCH PREDICTION EXAMPLE")
print("="*70)

batch_df = pd.DataFrame({
    "age_group": ["0-20", "41-60", "80+", "21-40"],
    "gender": ["Male", "Female", "Female", "Male"],
    "dosage_form": ["Tablet", "Injection", "Patch", "Capsule"],
    "drug_class": ["Antibiotic", "Chemotherapy", "NSAID", "Anticoagulant"],
    "country": ["US", "US", "CA", "UK"],
    "num_drugs": [1, 4, 2, 3],
    "prior_adr": [0, 1, 0, 1],
    "renal_impairment": [0, 0, 1, 0],
    "hepatic_impairment": [0, 1, 0, 0]
})

print("\nBatch input (4 patients):")
print(batch_df.to_string(index=False))

X_batch = preprocess_test_data(batch_df, le_dict)
dt_preds = dt_pipe.predict(X_batch)
lr_preds = lr_pipe.predict(X_batch)

results_df = pd.DataFrame({
    "Patient": [f"P{i+1}" for i in range(len(batch_df))],
    "Age": batch_df["age_group"],
    "Drug": batch_df["drug_class"],
    "DT_Pred": [SEVERITY_LABELS[p] for p in dt_preds],
    "LR_Pred": [SEVERITY_LABELS[p] for p in lr_preds],
})

print("\n\nBatch predictions:")
print(results_df.to_string(index=False))

print("\n" + "="*70)
print("  Testing complete!")
print("="*70)
