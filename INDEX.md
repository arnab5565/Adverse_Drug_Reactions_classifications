# 🎯 START HERE - ADR Testing Index

> **You asked:** "How to test a random given data for this project?"
>
> **Answer:** Everything is set up! Pick any method below. 👇

---

## ⚡ 30-Second Start

```bash
python adr_predictor.py
```

This trains models and shows 3 working examples. Done!

---

## 📚 Which File Should I Read?

### I want to test RIGHT NOW (2 min total)
1. Read: [TEST_CHEATSHEET.md](TEST_CHEATSHEET.md)
2. Run: `python adr_predictor.py`
3. Copy/paste examples

**Result:** You can test any patient data

---

### I want to understand everything (30 min total)
1. Read: [TESTING_SUMMARY.md](TESTING_SUMMARY.md) (overview)
2. Run: `python adr_predictor.py` (see it work)
3. Read: [TESTING_QUICKSTART.md](TESTING_QUICKSTART.md) (4 methods)
4. Read: [TESTING_GUIDE.md](TESTING_GUIDE.md) (deep dive)
5. Check: [test_examples.py](test_examples.py) (10 patterns)

**Result:** You're a testing expert

---

### I want to run existing code (1 min total)
Choose ONE:

**Option 1 - Simple wrapper (easiest):**
```python
from adr_predictor import ADRPredictor

p = ADRPredictor()
r = p.predict("41-60", "Female", "Injection", "Anticoagulant")
print(r['decision_tree']['severity_label'])
```

**Option 2 - Full demo:**
```bash
python test_single_prediction.py
```

**Option 3 - Code examples:**
```python
# Copy any example from test_examples.py
```

---

## 📋 Quick Reference

### 4 Ways to Test

| Way | File | Time | Best For |
|-----|------|------|----------|
| **Fastest** | `adr_predictor.py` | 30 sec | Quick tests |
| **Learning** | `TEST_CHEATSHEET.md` | 2 min | Getting started |
| **Complete** | `TESTING_GUIDE.md` | 15 min | Full understanding |
| **Patterns** | `test_examples.py` | 10 min | Copy & adapt |

### 3 Common Tests

```python
# Test 1: Low risk
p.predict("21-40", "Female", "Tablet", "Antibiotic")

# Test 2: High risk  
p.predict("80+", "Male", "Injection", "Chemotherapy", num_drugs=5, prior_adr=1)

# Test 3: Batch
p.predict_batch(pd.read_csv("patients.csv"))
```

### 4 Severity Classes

```
0: No Reaction (Low)
1: Hospitalization (Medium)
2: Life-Threatening/Disabling (High)
3: Death (Critical)
```

---

## 🚀 Choose Your Path

### Path 1: "Just show me working code!" ⚡⚡⚡
```bash
python adr_predictor.py
```
→ Trains models, tests 3 examples, batch predictions

---

### Path 2: "I need examples to copy/paste" ⚡⚡
```python
# Copy from test_examples.py
# 10 different patterns ready to use
```

---

### Path 3: "I want to understand everything" ⚡
Read in this order:
1. TESTING_SUMMARY.md (overview - 5 min)
2. TEST_CHEATSHEET.md (reference - 2 min)
3. TESTING_GUIDE.md (deep dive - 10 min)

---

## 📁 File Guide

```
⭐ START WITH THESE:
├── TEST_CHEATSHEET.md          Ultra-quick reference (2 min)
├── TESTING_SUMMARY.md          Complete overview (5 min)
├── adr_predictor.py            Simple wrapper (RUN THIS!)
│
📚 FOR LEARNING:
├── TESTING_QUICKSTART.md       Quick guide (5 min)
├── TESTING_GUIDE.md            Full documentation (15 min)
├── test_examples.py            10 code patterns
│
🔧 FOR TESTING:
├── adr_predictor.py            Run: python adr_predictor.py
├── test_single_prediction.py   Run: python test_single_prediction.py
│
📖 FOR REFERENCE:
├── README.md                   Project overview
├── This file (INDEX.md)        You are here
```

---

## 💡 Answer to Your Question

**"How to test a random given data for this project?"**

### The Short Answer
```python
from adr_predictor import ADRPredictor

p = ADRPredictor()
result = p.predict(age_group, gender, dosage_form, drug_class)
print(result['decision_tree']['severity_label'])
```

### The Complete Answer
See [TESTING_GUIDE.md](TESTING_GUIDE.md) for 4 different methods with examples.

### The Quick Demo
```bash
python adr_predictor.py
```

---

## ✅ What's Available

✅ Simple wrapper for testing (`adr_predictor.py`)
✅ Full testing script with examples (`test_single_prediction.py`)
✅ 10 code patterns ready to copy (`test_examples.py`)
✅ 4 documentation guides (5-15 min reads each)
✅ All models pre-trained and ready
✅ Real FAERS data integration
✅ Batch prediction support
✅ Confidence scores included

---

## 🎯 Next Step

Pick ONE:

1. **Want results now?** → Run `python adr_predictor.py`
2. **Want to learn?** → Read `TEST_CHEATSHEET.md`
3. **Want to code?** → Copy from `test_examples.py`
4. **Want everything?** → Read `TESTING_SUMMARY.md`

---

## 🐛 If Something Goes Wrong

1. Not installed? → `pip install -r requirements.txt`
2. Can't run? → `python adr_predictor.py`
3. Confused? → Read `TEST_CHEATSHEET.md`
4. Still stuck? → Check `TESTING_GUIDE.md` Troubleshooting section

---

## 📊 What You Can Do

- ✅ Test single patients
- ✅ Batch test from CSV
- ✅ Get confidence scores
- ✅ Compare two models
- ✅ Interactive input
- ✅ Integrate into apps

Examples for all of these are in `test_examples.py`

---

## 🎓 Learning Time Estimates

| Activity | Time |
|----------|------|
| Run demo | 30 sec |
| Read cheatsheet | 2 min |
| Understand one pattern | 5 min |
| Master all 10 patterns | 30 min |
| Read complete guide | 20 min |

Total to proficiency: **~30 minutes**

---

## 🚀 Quick Start (Right Now!)

```bash
# Step 1: Run demo
python adr_predictor.py

# Step 2: Read quick ref
# → Open TEST_CHEATSHEET.md

# Step 3: Test your data
python -c "
from adr_predictor import ADRPredictor
p = ADRPredictor()
r = p.predict('41-60', 'Female', 'Injection', 'Anticoagulant')
p.print_result(r)
"
```

Done! You can now test any patient data. 🎉

---

## 📞 Files Quick Links

- **Quick Reference:** [TEST_CHEATSHEET.md](TEST_CHEATSHEET.md)
- **Full Overview:** [TESTING_SUMMARY.md](TESTING_SUMMARY.md)
- **Quick Start:** [TESTING_QUICKSTART.md](TESTING_QUICKSTART.md)
- **Complete Guide:** [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Code Examples:** [test_examples.py](test_examples.py)
- **Simple Wrapper:** [adr_predictor.py](adr_predictor.py)

---

**Ready? Run this:** `python adr_predictor.py` 🚀
