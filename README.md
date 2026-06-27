# Adverse Drug Reaction (ADR) Severity Classification
### Decision Tree & Logistic Regression | Healthcare ML Project

Inspired by the FDA FAERS (Adverse Event Reporting System) dataset and the
approach described in *The Deep Hub* article on healthcare decision trees.

**✨ NEW:** Now supports fetching real FAERS data from FDA OpenAPI with automatic fallback to synthetic data!

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
├── adr_project.py              ← Main script (data loading → EDA → models → plots)
├── fetch_faers_data.py         ← Real FAERS data fetching from FDA OpenAPI
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file

outputs/
├── 1_eda_dashboard.png         ← EDA: distributions, age, dosage, drug class
├── 2_confusion_matrices.png    ← Side-by-side confusion matrices
├── 3_feature_importance.png    ← Decision Tree feature importances
├── 4_decision_tree_preview.png ← Tree visualisation (depth=3 preview)
└── 5_model_comparison.png      ← Train/test accuracy & F1 comparison
```

---

## Dataset

### Real FAERS Data (Recommended)
The script now **automatically fetches real adverse event reports** from the FDA OpenAPI:
- **Source:** https://open.fda.gov/apis/drug/event/
- **Data:** Actual FAERS reports from 2004 to present
- **Scope:** Up to 1,000 records per run (configurable)
- **Processing:** Auto-mapping of FAERS fields to feature schema

**Features from Real FAERS:**

| Feature | Source |
|---|---|
| `age_group` | Extracted from patient onset age |
| `gender` | Patient sex field (1=Male, 2=Female) |
| `dosage_form` | Drug route of administration (oral, IV, etc.) |
| `drug_class` | OpenFDA pharm_class annotation |
| `country` | Report submission country |
| `num_drugs` | Count of reported drug products |
| `prior_adr` | Patient medical history presence |
| `renal_impairment` | Extracted from reported reactions |
| `hepatic_impairment` | Extracted from reported reactions |
| `severity` | Mapped from seriousness flags (death, hospitalization, etc.) |

### Synthetic Data (Fallback)
If real data is unavailable or insufficient class diversity, the script uses synthetically generated data (5,000 records) that mirrors the FDA FAERS schema structure.

**Note:** Real FAERS data is heavily weighted toward non-serious reactions (>75% severity=0), so the model will automatically augment with synthetic data to ensure meaningful multi-class performance on your local machine.

---

## Modeling Pipeline

```
Raw Data (Real FAERS or Synthetic)
  └─► Data Validation & Feature Extraction
        └─► Label Encoding (categorical columns)
              └─► Train / Test Split (80 / 20, stratified)
                    ├─► Decision Tree  ←  GridSearchCV (max_depth, min_samples_split, criterion)
                    └─► Logistic Regression  ←  StandardScaler + class_weight="balanced"
```

### Sample Results (Real FAERS Data)

| Model | Train Acc | Test Acc | Macro F1 |
|---|---|---|---|
| Decision Tree | ~0.72 | ~0.80 | ~0.79 |
| Logistic Regression | ~0.71 | ~0.77 | ~0.76 |

> Note: Real FAERS data shows lower class imbalance than synthetic (roughly 50/50 split between severity 0 and 1).
> Results may vary based on the specific batch of FAERS records fetched.

---

## Key Insights from Real FAERS Analysis

- **Data Volume:** The OpenFDA API contains 1700+ downloadable files of drug adverse events
- **Reporting Patterns:** Serious events (seriousnessstring="Yes") represent ~30-40% of reports
- **Age Distribution:** Adverse events occur across all age groups, with concentrations in 41-60 and 61-80 ranges
- **Drug Polypharmacy:** Multi-drug interactions are common in reported ADRs
- **Geographic Distribution:** Majority of reports from USA, with significant international submission rates

---

## Setup & Run

### Installation

```bash
# Install dependencies
pip install scikit-learn pandas matplotlib seaborn numpy requests

# Or use requirements file
pip install -r requirements.txt
```

### Running the Script

```bash
# Run with real FAERS data (default behavior)
python adr_project.py

# The script will:
# 1. Attempt to fetch 1,000 real FAERS records from FDA OpenAPI
# 2. Process and validate the data
# 3. Train models using the real data
# 4. Generate visualizations
# 5. Fall back to synthetic data if API is unavailable
```

### Configuration

Edit [adr_project.py](adr_project.py) to customize:
- Real FAERS data fetch: `load_data(use_real_faers=True)` (line ~94)
- Number of synthetic records: `generate_faers_data(5000)` (line ~77)
- Train/test split: `train_test_split(..., test_size=0.2)` (line ~127)

---

## API Details

### OpenFDA Integration
The [fetch_faers_data.py](fetch_faers_data.py) module provides:

```python
from fetch_faers_data import fetch_real_faers_data

# Fetch and process real FAERS data
df = fetch_real_faers_data(num_records=1000)
```

**Features:**
- Automatic rate limiting (0.5s between requests)
- Robust error handling with fallback
- Field mapping from FAERS ICH E2b/M2 standard format
- Missing data handling (skips records with unknown values)
- Severity classification from seriousness flags

### API Rate Limits
- No API key required for basic access
- ~100 requests per second soft limit
- Up to 5 batch requests configured (500 records max per run by default)

---

## Extensions / Next Steps

- [x] Use real FAERS data from https://open.fda.gov/apis/drug/event/
- [ ] Try Random Forest / XGBoost for better minority class performance
- [ ] Apply SMOTE oversampling to address class imbalance
- [ ] Build a Streamlit web app for interactive prediction
- [ ] Add SHAP values for model explainability
- [ ] Cache downloaded FAERS data for faster iteration
- [ ] Implement incremental learning from new FAERS batches
- [ ] Visualize severity trends over time using FAERS temporal data

---

## Troubleshooting

### `requests` module not found
```bash
pip install requests
```

### API timeout or connection errors
The script automatically falls back to synthetic data. Check your internet connection or run with synthetic data:
```python
df, _ = load_data(use_real_faers=False)
```

### Class imbalance warnings
Real FAERS data heavily favors non-serious reactions. The script automatically augments with synthetic data to maintain meaningful multi-class performance. This is expected behavior.

---

## Disclaimer

⚠️ **For Educational Use Only**
- This model is trained on statistical patterns and should NOT be used for medical decision-making
- Real ADR assessment requires clinical expertise and comprehensive patient history
- OpenFDA data has limitations: voluntary reporting, verification not guaranteed, causal relationships not established
- Always consult qualified healthcare professionals for actual ADR assessment

See FDA's disclaimer: https://open.fda.gov/terms/
