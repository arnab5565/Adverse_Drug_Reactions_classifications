# 🚀 Getting Started in 60 Seconds

## What You Asked
"How to test a random given data for this project?"

## What You Get
✅ A simple Python function to predict ADR severity for any patient

## DO THIS NOW (60 seconds)

### Step 1: Run the demo (30 seconds)
```bash
python adr_predictor.py
```

You'll see 3 example predictions and batch processing.

### Step 2: Test your own patient (30 seconds)
```python
from adr_predictor import ADRPredictor

p = ADRPredictor()

# Replace these with your data
result = p.predict(
    age_group="41-60",      # Required: 0-20, 21-40, 41-60, 61-80, 80+
    gender="Female",        # Required: Female or Male
    dosage_form="Injection", # Required: Tablet, Injection, Capsule, etc.
    drug_class="Anticoagulant" # Required: Antibiotic, Anticoagulant, NSAID, etc.
)

# See the result
print(result['decision_tree']['severity_label'])
```

**Done!** You've tested your first patient. 🎉

---

## That's It!

Everything else is optional reference material.

### Need more info?
- Quick reference: Read [TEST_CHEATSHEET.md](TEST_CHEATSHEET.md) (2 min)
- Full guide: Read [TESTING_GUIDE.md](TESTING_GUIDE.md) (15 min)
- Code patterns: See [test_examples.py](test_examples.py) (10 patterns)

### Need help?
- Can't run? → Run `pip install -r requirements.txt`
- Questions? → Check [TESTING_GUIDE.md](TESTING_GUIDE.md) Troubleshooting
- Confused about data? → See Valid Values below

---

## Valid Input Values (Copy & Paste)

```python
age_group = ["0-20", "21-40", "41-60", "61-80", "80+"]
gender = ["Female", "Male"]
dosage_form = ["Tablet", "Injection", "Capsule", "Solution", "Patch", "Inhaler"]
drug_class = ["Antibiotic", "Anticoagulant", "NSAID", "Antihypertensive", "Chemotherapy", "Antidiabetic", "Antidepressant"]
```

---

## Output Meanings

```
"No Reaction" = Low risk ✓
"Hospitalization" = Medium risk ⚠
"Life-Threatening/Disabling" = High risk ⚠⚠
"Death" = Critical risk ⚠⚠⚠
```

---

## 3 Ways to Test

### Way 1: Single patient (most common)
```python
p = ADRPredictor()
r = p.predict("41-60", "Female", "Tablet", "NSAID")
print(r['decision_tree']['severity_label'])
```

### Way 2: Multiple patients from CSV
```python
import pandas as pd
patients = pd.read_csv("patients.csv")
results = p.predict_batch(patients)
```

### Way 3: Pretty formatted output
```python
result = p.predict("80+", "Male", "Injection", "Anticoagulant")
p.print_result(result)  # Shows both models + confidence
```

---

## Common Examples

```python
# Example 1: Low risk (young, oral, single drug)
p.predict("21-40", "Female", "Tablet", "Antibiotic")
# Output: "No Reaction"

# Example 2: High risk (elderly, IV, multiple drugs)
p.predict("80+", "Male", "Injection", "Chemotherapy")
# Output: "Life-Threatening/Disabling"

# Example 3: With more details
p.predict("61-80", "Female", "Injection", "Anticoagulant", 
          num_drugs=3, prior_adr=1, renal_impairment=1)
# Output: "Hospitalization" or higher
```

---

## Next Steps (Optional)

After running the demo:

1. **Want more examples?** → See [test_examples.py](test_examples.py)
2. **Want full documentation?** → See [TESTING_GUIDE.md](TESTING_GUIDE.md)
3. **Want batch processing?** → See Pattern 3 above
4. **Want to understand data format?** → See [TEST_CHEATSHEET.md](TEST_CHEATSHEET.md)

---

## That's Really It!

```bash
# Run this:
python adr_predictor.py

# Then copy/modify this:
from adr_predictor import ADRPredictor
p = ADRPredictor()
result = p.predict("41-60", "Female", "Tablet", "NSAID")
print(result['decision_tree']['severity_label'])
```

You're done! 🎉

---

## Questions?

| Question | Answer |
|----------|--------|
| How to test multiple patients? | Use `p.predict_batch(dataframe)` |
| How to get confidence scores? | They're in the result dict under `confidence` |
| How to use different drug class? | Use a value from the Valid Values list above |
| How to add more data? | Load CSV → `predict_batch()` → done |
| How to get both model predictions? | Results include both DT and LR automatically |

---

**Everything is ready. Just run `python adr_predictor.py` and you're set!** 🚀
