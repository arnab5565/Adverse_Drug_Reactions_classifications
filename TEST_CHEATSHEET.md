# ADR Model Testing - Ultra-Quick Reference

## Start Here (Copy & Paste)

```python
from adr_predictor import ADRPredictor

predictor = ADRPredictor()

# Test 1: Young person on antibiotic (low risk)
r1 = predictor.predict("21-40", "Female", "Tablet", "Antibiotic")
print(f"Young patient: {r1['decision_tree']['severity_label']}")

# Test 2: Elderly on blood thinner IV (high risk)
r2 = predictor.predict("80+", "Male", "Injection", "Anticoagulant", num_drugs=3, prior_adr=1)
print(f"Elderly patient: {r2['decision_tree']['severity_label']}")

# Test 3: Multiple patients
import pandas as pd
batch = pd.DataFrame({
    "age_group": ["41-60", "80+"],
    "gender": ["F", "M"],
    "dosage_form": ["Injection", "Tablet"],
    "drug_class": ["Anticoagulant", "NSAID"]
})
results = predictor.predict_batch(batch)
```

---

## One-Liners

```python
# Single prediction (minimum required args)
result = ADRPredictor().predict("41-60", "Female", "Tablet", "NSAID")

# Get just the class
severity = result['decision_tree']['severity_label']

# Get with confidence
predictor = ADRPredictor()
r = predictor.predict("80+", "Male", "Injection", "Chemotherapy")
print(f"{r['decision_tree']['severity_label']} ({r['decision_tree']['confidence']:.1%})")
```

---

## Valid Values (Quick Reference)

```python
age_group = ["0-20", "21-40", "41-60", "61-80", "80+"]
gender = ["Female", "Male"]
dosage_form = ["Tablet", "Injection", "Capsule", "Solution", "Patch", "Inhaler"]
drug_class = ["Antibiotic", "Anticoagulant", "NSAID", "Antihypertensive", 
              "Chemotherapy", "Antidiabetic", "Antidepressant"]
country = ["US", "CA", "UK", "DE", "Other"]  # Optional
```

---

## Output Classes

```
0 = "No Reaction" (Low risk)
1 = "Hospitalization" (Medium risk)
2 = "Life-Threatening/Disabling" (High risk)
3 = "Death" (Critical risk)
```

---

## Files to Use

| File | Purpose | Run Command |
|------|---------|-------------|
| `adr_predictor.py` | Simple wrapper | `python adr_predictor.py` |
| `test_single_prediction.py` | Full demo | `python test_single_prediction.py` |
| `TESTING_QUICKSTART.md` | Quick guide | Read it |
| `TESTING_GUIDE.md` | Full guide | Read it |

---

## Common Patterns

### Pattern 1: Simple Test
```python
from adr_predictor import ADRPredictor
p = ADRPredictor()
r = p.predict("41-60", "Female", "Tablet", "NSAID")
p.print_result(r)
```

### Pattern 2: CSV Input
```python
import pandas as pd
from adr_predictor import ADRPredictor
df = pd.read_csv("patients.csv")
p = ADRPredictor()
results = p.predict_batch(df)
```

### Pattern 3: Custom Logic
```python
from adr_predictor import ADRPredictor
p = ADRPredictor()

patients = [
    ("80+", "F", "Injection", "Anticoagulant"),
    ("21-40", "M", "Tablet", "Antibiotic"),
    ("60+", "F", "Patch", "NSAID")
]

for age, gender, form, drug in patients:
    r = p.predict(age, gender, form, drug)
    print(f"{age}: {r['decision_tree']['severity_label']}")
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Unknown category" | Use valid value from list above |
| "Shape mismatch" | Provide all 4 required args (age, gender, form, drug) |
| No confidence | Some models don't provide it - that's ok |
| Different predictions | Wrap in try/except or use higher confidence threshold |

---

## Examples in 30 Seconds

```bash
# Run demos
python adr_predictor.py

# Or test specific scenarios
python -c "
from adr_predictor import ADRPredictor
p = ADRPredictor()
# Low risk
print('Low risk:', p.predict('21-40', 'F', 'Tablet', 'Antibiotic')['decision_tree']['severity_label'])
# High risk  
print('High risk:', p.predict('80+', 'M', 'Injection', 'Chemotherapy')['decision_tree']['severity_label'])
"
```

---

## Minimal Example

```python
from adr_predictor import ADRPredictor

# Create predictor (trains models automatically)
p = ADRPredictor()

# Make prediction
result = p.predict(
    age_group="41-60",
    gender="Female", 
    dosage_form="Injection",
    drug_class="Anticoagulant"
)

# Print result
p.print_result(result)
```

That's it! 🎉
