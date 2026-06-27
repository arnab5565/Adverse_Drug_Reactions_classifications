# Testing ADR Model - Quick Start Guide

## 🚀 Fastest Way (30 seconds)

```bash
python adr_predictor.py
```

This trains models and shows 3 example predictions + batch predictions.

---

## 📊 Available Testing Methods

### 1. **Simple Wrapper** ⭐ (Easiest)
**File:** `adr_predictor.py`

```python
from adr_predictor import ADRPredictor

predictor = ADRPredictor()

# Single prediction
result = predictor.predict(
    age_group="41-60",
    gender="Female",
    dosage_form="Injection",
    drug_class="Anticoagulant",
    num_drugs=3
)

# Pretty print
predictor.print_result(result)
```

**Pros:** Minimal code, handles all preprocessing, confidence scores included
**Run:** `python adr_predictor.py`

---

### 2. **Full Testing Script**
**File:** `test_single_prediction.py`

```bash
python test_single_prediction.py
```

Trains fresh models and tests 3 risk levels + batch predictions.

**Features:**
- 3 example patients (low/medium/high risk)
- Batch prediction on 4 patients
- Detailed reporting with confidence scores

**Run:** `python test_single_prediction.py`

---

### 3. **Code Examples**
**File:** `test_examples.py`

Contains 10 different testing patterns:
1. Simple way
2. Single patient
3. Load from CSV
4. Valid input values
5. Batch predictions
6. With confidence scores
7. Interactive input
8. Troubleshooting
9. Model comparison
10. Real-world examples

**Run:** Copy/paste any example into your Python script

---

### 4. **Documentation**
**File:** `TESTING_GUIDE.md`

Comprehensive guide covering:
- All 4 testing methods
- Data format reference
- Preprocessing steps
- Common errors & solutions
- Example test cases

---

## 🎯 Choose by Your Use Case

| Use Case | Recommended Method | Command |
|----------|-------------------|---------|
| Quick test | Simple Wrapper | `python adr_predictor.py` |
| Full demo | Testing Script | `python test_single_prediction.py` |
| Learning | Code Examples | Read `test_examples.py` |
| Deep dive | Documentation | Read `TESTING_GUIDE.md` |
| Integration | Custom | Adapt from test_examples.py |

---

## 💡 Most Common Testing Patterns

### Pattern 1: Test Single Patient
```python
from adr_predictor import ADRPredictor

predictor = ADRPredictor()
result = predictor.predict("80+", "Female", "Tablet", "NSAID")
print(result)
```

### Pattern 2: Test Multiple Patients
```python
import pandas as pd
from adr_predictor import ADRPredictor

predictor = ADRPredictor()

batch = pd.DataFrame({
    "age_group": ["41-60", "80+", "21-40"],
    "gender": ["F", "F", "M"],
    "dosage_form": ["Injection", "Tablet", "Capsule"],
    "drug_class": ["Anticoagulant", "NSAID", "Antibiotic"]
})

results = predictor.predict_batch(batch)
```

### Pattern 3: Test from CSV
```python
import pandas as pd
from adr_predictor import ADRPredictor

predictor = ADRPredictor()
data = pd.read_csv("patients.csv")
results = predictor.predict_batch(data)
```

### Pattern 4: Interactive Input
```python
from adr_predictor import ADRPredictor

predictor = ADRPredictor()

age = input("Age: ")
gender = input("Gender: ")
drug = input("Drug: ")
# ... collect other inputs

result = predictor.predict(age, gender, drug, ...)
predictor.print_result(result)
```

---

## 📋 Data Format

### Required Inputs (10 fields):
1. `age_group` - "0-20", "21-40", "41-60", "61-80", "80+"
2. `gender` - "Female", "Male"
3. `dosage_form` - "Tablet", "Injection", "Capsule", "Solution", "Patch", "Inhaler"
4. `drug_class` - "Antibiotic", "Anticoagulant", "NSAID", etc.
5. `country` - "US", "CA", "UK", "DE", "Other" (optional, default: "US")
6. `num_drugs` - 1-8 (optional, default: 1)
7. `prior_adr` - 0 or 1 (optional, default: 0)
8. `renal_impairment` - 0 or 1 (optional, default: 0)
9. `hepatic_impairment` - 0 or 1 (optional, default: 0)

### Output:
```python
{
    "patient": {...},  # Echo of input data
    "decision_tree": {
        "severity_class": 1,
        "severity_label": "Hospitalization",
        "confidence": 0.714
    },
    "logistic_regression": {
        "severity_class": 1,
        "severity_label": "Hospitalization",
        "confidence": 0.352
    }
}
```

---

## 🔍 Severity Classes

| Class | Label | Risk Level |
|-------|-------|-----------|
| 0 | No Reaction | Low ✓ |
| 1 | Hospitalization | Medium ⚠ |
| 2 | Life-Threatening/Disabling | High ⚠⚠ |
| 3 | Death | Critical ⚠⚠⚠ |

---

## 📁 Project Files

```
d:\Resume\ADR\
├── adr_project.py              Main script (uses real FAERS data)
├── adr_predictor.py            ⭐ Simple wrapper - START HERE
├── test_single_prediction.py   Full testing demo
├── test_examples.py            Code examples (10 patterns)
├── fetch_faers_data.py         Real FAERS data fetching
├── TESTING_GUIDE.md            Complete documentation
├── README.md                   Project overview
├── requirements.txt            Dependencies
└── outputs/                    Generated visualizations
    ├── 1_eda_dashboard.png
    ├── 2_confusion_matrices.png
    ├── 3_feature_importance.png
    ├── 4_decision_tree_preview.png
    └── 5_model_comparison.png
```

---

## ✅ Quick Checklist

- [ ] Installed requirements: `pip install -r requirements.txt`
- [ ] Run simple test: `python adr_predictor.py`
- [ ] Understand data format (10 fields)
- [ ] Know the 4 severity classes
- [ ] Check valid values for each field
- [ ] Read error messages if prediction fails

---

## 🐛 Common Issues

### "Unknown value in column X"
Use only values from the reference table. Check [test_examples.py](test_examples.py#L62-L72).

### "Shape mismatch"
Ensure all 9 features are provided. The wrapper handles this automatically.

### "Confidence is None"
Some model types don't support probability estimates. This is normal.

### "Feature names don't match"
Use the wrapper (`adr_predictor.py`) which handles this automatically.

---

## 🎓 Learning Path

1. **Start:** `python adr_predictor.py` (30 seconds)
2. **Read:** `TESTING_GUIDE.md` (5 minutes)
3. **Learn:** `test_examples.py` (10 patterns)
4. **Customize:** Copy/modify for your use case
5. **Deploy:** Use wrapper in your application

---

## 📞 Support

For detailed information, see:
- **Testing:** [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Examples:** [test_examples.py](test_examples.py)
- **Code:** [adr_predictor.py](adr_predictor.py) (well commented)
- **Project:** [README.md](README.md)

Happy testing! 🚀
