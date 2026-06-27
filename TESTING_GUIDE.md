# Testing Your ADR Model - Complete Guide

## Overview
This guide shows 4 different ways to test the trained ADR severity prediction models on new patient data.

---

## Method 1: Use the Provided Testing Script ⭐ (Recommended)

**File:** [test_single_prediction.py](test_single_prediction.py)

### Run it:
```bash
python test_single_prediction.py
```

### What it does:
- Trains fresh models on synthetic data
- Tests 3 example patients with different risk profiles
- Performs batch predictions on 4 patients
- Shows confidence scores for each prediction

### Key Functions:

```python
# Create a single test record
patient = create_test_record(
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

# Preprocess for model
X_test = preprocess_test_data(patient, le_dict)

# Get predictions with confidence
dt_pred, dt_conf = get_prediction_confidence(dt_pipe, X_test)
lr_pred, lr_conf = get_prediction_confidence(lr_pipe, X_test)
```

---

## Method 2: Quick Python Script (Inline)

Create a simple test script:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 1. Create your test data
test_data = pd.DataFrame({
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

print("Test Data:")
print(test_data)

# 2. Encode categorical features (same as training)
le_dict = {
    "age_group": LabelEncoder(),
    "gender": LabelEncoder(),
    "dosage_form": LabelEncoder(),
    "drug_class": LabelEncoder(),
    "country": LabelEncoder(),
}

for col in le_dict:
    le_dict[col].fit(["0-20", "21-40", "41-60", "61-80", "80+"] if col == "age_group"
                     else ["Female", "Male"] if col == "gender"
                     else ["Capsule", "Inhaler", "Injection", "Patch", "Solution", "Tablet", "Unknown"] if col == "dosage_form"
                     else ["Antibiotic", "Antidepressant", "Antidiabetic", "Anticoagulant", "Antihypertensive", "Chemotherapy", "NSAID", "Other", "Statin"] if col == "drug_class"
                     else ["CA", "DE", "GB", "US", "Other"])
    test_data[col] = le_dict[col].transform(test_data[col])

# 3. Make prediction
X_test = test_data.values
prediction = model.predict(X_test)
confidence = model.predict_proba(X_test).max()

SEVERITY_LABELS = ["No Reaction", "Hospitalization", "Life-Threatening/Disabling", "Death"]
print(f"\nPrediction: {SEVERITY_LABELS[prediction[0]]}")
print(f"Confidence: {confidence:.1%}")
```

---

## Method 3: Load from CSV File

### Create a CSV with test data:

**patients.csv:**
```csv
age_group,gender,dosage_form,drug_class,country,num_drugs,prior_adr,renal_impairment,hepatic_impairment
41-60,Female,Injection,Anticoagulant,US,3,1,0,1
21-40,Male,Tablet,Antibiotic,CA,1,0,0,0
80+,Female,Patch,NSAID,UK,2,0,1,0
61-80,Male,Solution,Chemotherapy,US,4,1,1,1
```

### Load and predict:

```python
import pandas as pd
from test_single_prediction import preprocess_test_data

# Load test data
test_df = pd.read_csv("patients.csv")

# Preprocess and predict
X_test = preprocess_test_data(test_df, le_dict)
predictions_dt = dt_pipe.predict(X_test)
predictions_lr = lr_pipe.predict(X_test)

# Add predictions to original data
test_df["DT_Prediction"] = [SEVERITY_LABELS[p] for p in predictions_dt]
test_df["LR_Prediction"] = [SEVERITY_LABELS[p] for p in predictions_lr]

print(test_df)
test_df.to_csv("predictions.csv", index=False)
```

---

## Method 4: Interactive Input

```python
# Interactive input from user
print("\n=== ADR SEVERITY PREDICTOR ===\n")

age_group = input("Age group [0-20/21-40/41-60/61-80/80+]: ")
gender = input("Gender [Female/Male]: ")
dosage_form = input("Dosage form [Tablet/Injection/Capsule/Solution/Patch/Inhaler]: ")
drug_class = input("Drug class [Antibiotic/Anticoagulant/NSAID/Antihypertensive/Chemotherapy/etc]: ")
country = input("Country [US/CA/UK/DE/Other]: ")
num_drugs = int(input("Number of concurrent drugs [1-8]: "))
prior_adr = int(input("Prior ADR history [0/1]: "))
renal_imp = int(input("Renal impairment [0/1]: "))
hepatic_imp = int(input("Hepatic impairment [0/1]: "))

# Create record
patient = pd.DataFrame({
    "age_group": [age_group],
    "gender": [gender],
    "dosage_form": [dosage_form],
    "drug_class": [drug_class],
    "country": [country],
    "num_drugs": [num_drugs],
    "prior_adr": [prior_adr],
    "renal_impairment": [renal_imp],
    "hepatic_impairment": [hepatic_imp]
})

# Predict
X_test = preprocess_test_data(patient, le_dict)
pred = dt_pipe.predict(X_test)[0]
conf = dt_pipe.predict_proba(X_test).max()

print(f"\n✓ Predicted Severity: {SEVERITY_LABELS[pred]}")
print(f"✓ Confidence: {conf:.1%}")
```

---

## Data Format Reference

### Required Features (10 features):

| Feature | Valid Values | Type |
|---------|--------------|------|
| `age_group` | 0-20, 21-40, 41-60, 61-80, 80+ | string |
| `gender` | Female, Male | string |
| `dosage_form` | Tablet, Injection, Capsule, Solution, Patch, Inhaler, Unknown | string |
| `drug_class` | Antibiotic, Anticoagulant, NSAID, Antihypertensive, Chemotherapy, Antidiabetic, Antidepressant, Other, Statin | string |
| `country` | US, CA, UK, DE, Other | string |
| `num_drugs` | 1-8 | integer |
| `prior_adr` | 0 or 1 | integer |
| `renal_impairment` | 0 or 1 | integer |
| `hepatic_impairment` | 0 or 1 | integer |

### Output Classes:

| Class | Label |
|-------|-------|
| 0 | No Reaction |
| 1 | Hospitalization |
| 2 | Life-Threatening / Disabling |
| 3 | Death |

---

## Preprocessing Steps (Must Match Training)

1. **Label Encoding** - Convert categorical strings to integers
   ```
   age_group: 0-20→0, 21-40→1, 41-60→2, 61-80→3, 80+→4
   gender: Female→0, Male→1
   ... etc
   ```

2. **Scaling** (Only for Logistic Regression)
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_train)
   ```

3. **Order Matters** - Features must be in this order:
   ```
   [age_group, gender, dosage_form, drug_class, country, 
    num_drugs, prior_adr, renal_impairment, hepatic_impairment]
   ```

---

## Common Errors & Solutions

### Error: "Unknown category in column 'drug_class'"
```python
# Problem: Test data has drug class not in training data
# Solution: Use only valid categories from reference table
drug_class = "Other"  # Use "Other" as fallback
```

### Error: "Shape mismatch - Expected 9 features, got X"
```python
# Problem: Wrong number of input features
# Check: All 9 features present and in correct order
# Should be: [age_group, gender, dosage_form, drug_class, country, 
#            num_drugs, prior_adr, renal_impairment, hepatic_impairment]
```

### Error: "Confidence is None"
```python
# Problem: Model doesn't support predict_proba
# Solution: Check model type and use appropriate method
if hasattr(model, 'predict_proba'):
    confidence = model.predict_proba(X_test).max()
else:
    confidence = None  # Some models don't provide confidence
```

---

## Example Test Cases

### Test Case 1: Low Risk Patient
```python
patient_low = create_test_record(
    age_group="25",  # Young
    gender="Female",
    dosage_form="Capsule",  # Oral
    drug_class="Antibiotic",  # Common, low-risk
    num_drugs=1,  # Monotherapy
    prior_adr=0,  # No history
    renal_impairment=0,
    hepatic_impairment=0
)
# Expected: Class 0 (No Reaction) - High confidence
```

### Test Case 2: Medium Risk Patient
```python
patient_med = create_test_record(
    age_group="60",  # Older
    gender="Male",
    dosage_form="Tablet",
    drug_class="NSAID",
    country="US",
    num_drugs=2,  # Some polypharmacy
    prior_adr=1,  # Previous reaction
    renal_impairment=1,
    hepatic_impairment=0
)
# Expected: Class 1 (Hospitalization) - Medium confidence
```

### Test Case 3: High Risk Patient
```python
patient_high = create_test_record(
    age_group="75",  # Very old
    gender="Female",
    dosage_form="Injection",  # IV route
    drug_class="Chemotherapy",  # Severe class
    country="US",
    num_drugs=5,  # High polypharmacy
    prior_adr=1,
    renal_impairment=1,
    hepatic_impairment=1
)
# Expected: Class 3 (Death) or Class 2 (Life-threatening) - High confidence
```

---

## Batch Processing

### Predict on Multiple Records:
```python
# Create batch
batch = pd.DataFrame({
    "age_group": ["41-60", "21-40", "80+"],
    "gender": ["Female", "Male", "Female"],
    "dosage_form": ["Injection", "Tablet", "Patch"],
    "drug_class": ["Anticoagulant", "Antibiotic", "NSAID"],
    "country": ["US", "CA", "UK"],
    "num_drugs": [3, 1, 2],
    "prior_adr": [1, 0, 0],
    "renal_impairment": [0, 0, 1],
    "hepatic_impairment": [1, 0, 0]
})

# Predict all at once
X_batch = preprocess_test_data(batch, le_dict)
predictions = dt_pipe.predict(X_batch)

# Add to dataframe
batch["prediction"] = [SEVERITY_LABELS[p] for p in predictions]
print(batch)
```

---

## Performance Notes

- **Decision Tree**: Faster inference, more interpretable
- **Logistic Regression**: Slower, better calibrated probabilities
- **Consensus**: When both models agree, prediction is more reliable
- **Disagreement**: When models disagree, use confidence scores to decide

---

## Next Steps

1. Run `python test_single_prediction.py` to see examples
2. Modify the examples to test your own scenarios
3. Create a CSV file with your test data
4. Load and batch predict using Method 3
5. Build a web app (Streamlit/Flask) for interactive use
