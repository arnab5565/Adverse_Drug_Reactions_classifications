# Adverse Drug Reaction (ADR) Severity Classification
### Decision Tree & Logistic Regression | Healthcare ML Project

Inspired by the FDA FAERS (Adverse Event Reporting System) dataset and the
approach described in *The Deep Hub* article on healthcare decision trees.

---

## Problem Statement

Classify the **severity of an adverse drug reaction (ADR)** into one of four classes:

| Label | Class |
|-------|-------|
| 0 | No Reaction |
| 1 | Hospitalization |
| 2 | Life-Threatening / Disabling |
| 3 | Death |

---

## Project Structure

```
adr_classification/
├── adr_project.py       ← Main script (data gen → EDA → models → plots)
└── README.md

outputs/
├── 1_eda_dashboard.png        ← EDA: distributions, age, dosage, drug class
├── 2_confusion_matrices.png   ← Side-by-side confusion matrices
├── 3_feature_importance.png   ← Decision Tree feature importances
├── 4_decision_tree_preview.png ← Tree visualisation (depth=3 preview)
└── 5_model_comparison.png     ← Train/test accuracy & F1 comparison
```

---

## Dataset

Synthetically generated (5,000 records) to mirror the FDA FAERS schema.
Real FAERS data: https://open.fda.gov/data/downloads/

**Features:**

| Feature | Description |
|---|---|
| `age_group` | Patient age bin (0-20, 21-40, 41-60, 61-80, 80+) |
| `gender` | Female / Male |
| `dosage_form` | Tablet, Injection, Capsule, Solution, Patch, Inhaler |
| `drug_class` | Antibiotic, Anticoagulant, NSAID, Antihypertensive, … |
| `country` | USA, Canada, UK, Germany, Other |
| `num_drugs` | Number of concurrent medications (polypharmacy) |
| `prior_adr` | Prior adverse reaction history (0/1) |
| `renal_impairment` | Kidney impairment flag (0/1) |
| `hepatic_impairment` | Liver impairment flag (0/1) |

---

## Modeling Pipeline

```
Raw Data
  └─► Label Encoding (categorical columns)
        └─► Train / Test Split (80 / 20, stratified)
              ├─► Decision Tree  ←  GridSearchCV (max_depth, min_samples_split, criterion)
              └─► Logistic Regression  ←  StandardScaler + class_weight="balanced"
```

### Results

| Model | Train Acc | Test Acc | Macro F1 |
|---|---|---|---|
| Decision Tree | ~0.72 | ~0.60 | ~0.49 |
| Logistic Regression | ~0.51 | ~0.51 | ~0.43 |

> Decision Tree outperforms LR on this non-linear, imbalanced dataset.
> F1-macro is the primary metric (imbalanced classes).
> Mirrors the article's findings: majority classes (No Reaction, Hospitalization)
> predict well; minority classes (Life-Threatening, Death) are harder.

---

## Key Insights

- **Drug class** and **polypharmacy (num_drugs)** are the top features.
- **Injection** dosage form correlates with higher hospitalization (patient already in-hospital).
- **Tablets** correlate with more deaths (delayed reaction detection outside hospital).
- **Elderly patients (80+)** and those with **prior ADR history** show elevated severity.
- Class imbalance (50% No Reaction) → use F1-macro over raw accuracy.

---

## Setup & Run

```bash
pip install scikit-learn pandas matplotlib seaborn numpy

python adr_project.py
```

---

## Extensions / Next Steps

- [ ] Use real FAERS data from https://open.fda.gov/data/downloads/
- [ ] Try Random Forest / XGBoost for better minority class performance
- [ ] Apply SMOTE oversampling to address class imbalance
- [ ] Build a Streamlit web app for interactive prediction
- [ ] Add SHAP values for model explainability
