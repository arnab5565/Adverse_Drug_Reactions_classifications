"""
=============================================================================
Adverse Drug Reaction (ADR) Severity Classification
Decision Tree + Logistic Regression | Healthcare ML Project
=============================================================================


Dataset: Synthetically generated to mirror FDA FAERS structure
Target:  Multi-class ADR severity
         0 = No Reaction
         1 = Hospitalization
         2 = Life-Threatening / Disabling
         3 = Death
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Import real FAERS data fetcher
try:
    from fetch_faers_data import fetch_real_faers_data
    HAS_FAERS_FETCHER = True
except ImportError:
    HAS_FAERS_FETCHER = False

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4a",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#c0c0c0",
    "ytick.color":      "#c0c0c0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4a",
    "font.family":      "DejaVu Sans",
})
PALETTE = ["#4e8cff", "#ff6b6b", "#ffd166", "#06d6a0"]
SEVERITY_LABELS = ["No Reaction", "Hospitalization", "Life-Threatening/Disabling", "Death"]

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING  (Real FAERS data with synthetic fallback)
# ═══════════════════════════════════════════════════════════════════════════
def generate_faers_data(n=5000):
    """Generate synthetic FAERS data (fallback if real data unavailable)"""
    age_bins   = ["0-20", "21-40", "41-60", "61-80", "80+"]
    age_probs  = [0.06, 0.18, 0.32, 0.30, 0.14]
    genders    = ["Female", "Male"]
    dosage_forms = ["Tablet", "Injection", "Capsule", "Solution", "Patch", "Inhaler"]
    drug_classes = ["Antibiotic", "Anticoagulant", "NSAID", "Antihypertensive",
                    "Chemotherapy", "Antidiabetic", "Antidepressant"]
    countries  = ["USA", "Canada", "UK", "Germany", "Other"]
    country_probs = [0.55, 0.27, 0.08, 0.05, 0.05]

    age       = np.random.choice(age_bins, n, p=age_probs)
    gender    = np.random.choice(genders,  n, p=[0.56, 0.44])
    dosage    = np.random.choice(dosage_forms, n, p=[0.35, 0.25, 0.18, 0.10, 0.07, 0.05])
    drug_cls  = np.random.choice(drug_classes, n)
    country   = np.random.choice(countries, n, p=country_probs)
    num_drugs = np.random.randint(1, 8, n)          # polypharmacy count
    prior_adr = np.random.randint(0, 2, n)          # prior adverse reaction history
    renal_imp = np.random.choice([0, 1], n, p=[0.75, 0.25])
    hepatic_imp = np.random.choice([0, 1], n, p=[0.80, 0.20])

    # ── Build severity with realistic correlations ────────────────────────
    score = np.zeros(n)
    score += np.where(dosage == "Injection", 1.2, 0)
    score += np.where(dosage == "Tablet",    0.5, 0)
    score += np.where(drug_cls == "Chemotherapy",  1.5, 0)
    score += np.where(drug_cls == "Anticoagulant", 1.0, 0)
    score += np.where(drug_cls == "Antibiotic",    0.4, 0)
    score += np.where(age == "80+",   1.2, 0)
    score += np.where(age == "61-80", 0.7, 0)
    score += np.where(age == "0-20",  0.3, 0)
    score += num_drugs * 0.15
    score += prior_adr * 0.8
    score += renal_imp * 0.9
    score += hepatic_imp * 0.7
    score += np.random.normal(0, 0.5, n)            # noise

    # Bin score → 4 severity classes (imbalanced, like real FAERS)
    thresholds = np.percentile(score, [50, 72, 88])
    severity   = np.digitize(score, thresholds)     # 0,1,2,3

    df = pd.DataFrame({
        "age_group":        age,
        "gender":           gender,
        "dosage_form":      dosage,
        "drug_class":       drug_cls,
        "country":          country,
        "num_drugs":        num_drugs,
        "prior_adr":        prior_adr,
        "renal_impairment": renal_imp,
        "hepatic_impairment": hepatic_imp,
        "severity":         severity,
    })
    return df

def load_data(use_real_faers=True):
    """Load real FAERS data or fallback to synthetic"""
    data_source = "synthetic"
    
    if use_real_faers and HAS_FAERS_FETCHER:
        print("  Attempting to fetch real FAERS data from FDA OpenAPI...")
        df = fetch_real_faers_data(num_records=1000)
        if df is not None and not df.empty:
            # Check if we have meaningful class distribution
            severity_counts = df["severity"].value_counts()
            if len(severity_counts) >= 2:  # At least 2 classes
                print(f"  ✓ Real FAERS data loaded with {len(severity_counts)} severity classes")
                data_source = "real FAERS"
                return df, data_source
            else:
                print(f"  ⚠ Real data has limited class diversity ({len(severity_counts)} class). Augmenting with synthetic...")
                # Augment real data with synthetic to ensure class balance
                synthetic_df = generate_faers_data(min(5000, max(len(df) * 5, 2000)))
                df = pd.concat([df, synthetic_df], ignore_index=True)
                data_source = "real FAERS (augmented)"
                return df, data_source
        print("  Real data unavailable. Falling back to synthetic data.")
    
    df = generate_faers_data(5000)
    return df, data_source

print("=" * 65)
print("  ADR Severity Classification  |  Healthcare ML Project")
print("=" * 65)

df, data_source = load_data(use_real_faers=True)
data_type_label = "real FAERS" if data_source == "real FAERS" else "synthetic"
print(f"\n[1] Dataset loaded ({data_type_label})  →  {df.shape[0]:,} records, {df.shape[1]} features")
print(df.head())

# ═══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
cat_cols = ["age_group", "gender", "dosage_form", "drug_class", "country"]
df_enc   = df.copy()
le_dict  = {}
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df_enc.drop("severity", axis=1)
y = df_enc["severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[2] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"    Class distribution (train):\n    {y_train.value_counts().sort_index().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. DECISION TREE  (with GridSearchCV)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] Grid-searching Decision Tree hyperparameters …")
dt_pipe = Pipeline([("clf", DecisionTreeClassifier(random_state=42))])
dt_params = {
    "clf__max_depth":        [8, 12, 16, 20],
    "clf__min_samples_split":[0.005, 0.01, 0.02],
    "clf__criterion":        ["gini", "entropy"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt_gs = GridSearchCV(dt_pipe, dt_params, cv=cv, scoring="f1_macro", n_jobs=-1)
dt_gs.fit(X_train, y_train)

best_dt  = dt_gs.best_estimator_
y_pred_dt = best_dt.predict(X_test)
dt_train_acc = best_dt.score(X_train, y_train)
dt_test_acc  = accuracy_score(y_test, y_pred_dt)
dt_f1        = f1_score(y_test, y_pred_dt, average="macro", zero_division=0)

print(f"    Best params : {dt_gs.best_params_}")
print(f"    Train acc   : {dt_train_acc:.3f}   Test acc : {dt_test_acc:.3f}")
print(f"    Macro F1    : {dt_f1:.3f}")
print("\n    Classification Report (Decision Tree):")

# Get only the labels that exist in test set
unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred_dt])))
active_labels = [SEVERITY_LABELS[i] for i in unique_classes]
print(classification_report(y_test, y_pred_dt, target_names=active_labels, labels=unique_classes))

# ═══════════════════════════════════════════════════════════════════════════
# 4. LOGISTIC REGRESSION  (baseline comparison)
# ═══════════════════════════════════════════════════════════════════════════
print("[4] Training Logistic Regression …")
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight="balanced")),
])
lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)
lr_train_acc = lr_pipe.score(X_train, y_train)
lr_test_acc  = accuracy_score(y_test, y_pred_lr)
lr_f1        = f1_score(y_test, y_pred_lr, average="macro", zero_division=0)
print(f"    Train acc : {lr_train_acc:.3f}   Test acc : {lr_test_acc:.3f}")
print(f"    Macro F1  : {lr_f1:.3f}")
print("\n    Classification Report (Logistic Regression):")

# Get only the labels that exist in test set
unique_classes_lr = sorted(np.unique(np.concatenate([y_test, y_pred_lr])))
active_labels_lr = [SEVERITY_LABELS[i] for i in unique_classes_lr]
print(classification_report(y_test, y_pred_lr, target_names=active_labels_lr, labels=unique_classes_lr))

# ═══════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] Generating visualisations …")

# Get actual severity classes present in data
actual_severity_classes = sorted(df["severity"].unique())
actual_severity_labels = [SEVERITY_LABELS[i] for i in actual_severity_classes]
actual_colors = [PALETTE[i] for i in actual_severity_classes]

# ── Fig 1: EDA dashboard ──────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle("Adverse Drug Reaction — Exploratory Data Analysis",
              fontsize=16, fontweight="bold", color="#e0e0e0", y=1.01)

# (a) Severity class distribution
sev_counts = df["severity"].value_counts().sort_index()
bars = axes[0, 0].bar(actual_severity_labels, sev_counts.values, color=actual_colors, edgecolor="none")
axes[0, 0].set_title("Severity Class Distribution", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Severity Class")
axes[0, 0].set_ylabel("Count")
for bar, val in zip(bars, sev_counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 30, f"{val:,}",
                    ha="center", va="bottom", fontsize=9, color="#c0c0c0")
axes[0, 0].set_xticklabels(actual_severity_labels, rotation=15, ha="right", fontsize=8)

# (b) Age group distribution by severity
age_order = ["0-20", "21-40", "41-60", "61-80", "80+"]
age_sev   = df.groupby(["age_group", "severity"]).size().unstack(fill_value=0)
age_sev   = age_sev.reindex(age_order)
age_sev.plot(kind="bar", stacked=True, ax=axes[0, 1],
             color=actual_colors, edgecolor="none", legend=False)
axes[0, 1].set_title("Age Group × Severity", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Age Group")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_xticklabels(age_order, rotation=0)
leg = axes[0, 1].legend(actual_severity_labels, title="Severity",
                         fontsize=7, title_fontsize=8,
                         loc="upper right", framealpha=0.4)

# (c) Dosage form vs severity heatmap
dosage_sev = df.groupby(["dosage_form", "severity"]).size().unstack(fill_value=0)
dosage_sev.columns = [actual_severity_labels[i] for i in range(len(dosage_sev.columns))]
sns.heatmap(dosage_sev, ax=axes[0, 2], cmap="YlOrRd",
            linewidths=0.5, linecolor="#0f1117",
            annot=True, fmt="d", annot_kws={"size": 8},
            cbar_kws={"shrink": 0.8})
axes[0, 2].set_title("Dosage Form × Severity", fontsize=12, fontweight="bold")
axes[0, 2].set_xlabel("Severity Class")
axes[0, 2].set_ylabel("Dosage Form")
axes[0, 2].tick_params(axis="x", rotation=20)

# (d) Drug class vs severity
drug_sev = df.groupby(["drug_class", "severity"]).size().unstack(fill_value=0)
drug_sev.plot(kind="barh", stacked=True, ax=axes[1, 0],
              color=actual_colors, edgecolor="none", legend=False)
axes[1, 0].set_title("Drug Class × Severity", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Count")
axes[1, 0].set_ylabel("Drug Class")

# (e) Number of concurrent drugs distribution
axes[1, 1].hist(df["num_drugs"], bins=7, color="#4e8cff", edgecolor="#0f1117", alpha=0.85)
axes[1, 1].set_title("Polypharmacy (# Concurrent Drugs)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Number of Drugs")
axes[1, 1].set_ylabel("Frequency")

# (f) Gender split
gender_sev = df.groupby(["gender", "severity"]).size().unstack(fill_value=0)
gender_sev.plot(kind="bar", ax=axes[1, 2], color=actual_colors,
                edgecolor="none", legend=False)
axes[1, 2].set_title("Gender × Severity", fontsize=12, fontweight="bold")
axes[1, 2].set_xlabel("Gender")
axes[1, 2].set_ylabel("Count")
axes[1, 2].set_xticklabels(["Female", "Male"], rotation=0)

plt.tight_layout()
fig1.savefig(os.path.join("outputs", "1_eda_dashboard.png"),
             dpi=150, bbox_inches="tight", facecolor="#0f1117")
print("    Saved: 1_eda_dashboard.png")

# ── Fig 2: Confusion matrices side by side ────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Confusion Matrices — Multi-class ADR Severity",
              fontsize=14, fontweight="bold", color="#e0e0e0")

for ax, y_pred, title in zip(
    axes2,
    [y_pred_dt, y_pred_lr],
    ["Decision Tree (GridSearchCV)", "Logistic Regression"]
):
    cm = confusion_matrix(y_test, y_pred, labels=actual_severity_classes)
    im = ax.imshow(cm, cmap="Blues")
    n_classes = len(actual_severity_classes)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(actual_severity_labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels(actual_severity_labels, fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#333",
                    fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig2.savefig(os.path.join("outputs", "2_confusion_matrices.png"),
             dpi=150, bbox_inches="tight", facecolor="#0f1117")
print("    Saved: 2_confusion_matrices.png")

# ── Fig 3: Feature importance ─────────────────────────────────────────────
dt_clf    = best_dt.named_steps["clf"]
importances = pd.Series(dt_clf.feature_importances_, index=X.columns).sort_values()

fig3, ax3 = plt.subplots(figsize=(9, 5))
colors = [PALETTE[int(v / importances.max() * 3)] for v in importances]
importances.plot(kind="barh", ax=ax3, color=colors, edgecolor="none")
ax3.set_title("Feature Importances — Decision Tree", fontsize=13, fontweight="bold")
ax3.set_xlabel("Gini Importance")
ax3.set_ylabel("Feature")
ax3.axvline(importances.mean(), color="#ffd166", linestyle="--",
            linewidth=1, label=f"Mean ({importances.mean():.3f})")
ax3.legend(fontsize=9)
plt.tight_layout()
fig3.savefig(os.path.join("outputs", "3_feature_importance.png"),
             dpi=150, bbox_inches="tight", facecolor="#0f1117")
print("    Saved: 3_feature_importance.png")

# ── Fig 4: Decision Tree (trimmed for readability) ────────────────────────
fig4, ax4 = plt.subplots(figsize=(22, 9), facecolor="#0f1117")
plot_tree(
    dt_clf,
    max_depth=3,
    feature_names=X.columns.tolist(),
    class_names=["No Rxn", "Hosp", "Life-Threat", "Death"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax4,
    impurity=True,
)
ax4.set_title("Decision Tree (depth=3 preview — full tree max_depth=20)",
              fontsize=11, fontweight="bold", color="#e0e0e0", pad=10)
fig4.savefig(os.path.join("outputs", "4_decision_tree_preview.png"),
             dpi=120, bbox_inches="tight", facecolor="#0f1117")
print("    Saved: 4_decision_tree_preview.png")

# ── Fig 5: Model comparison bar chart ─────────────────────────────────────
models = ["Decision Tree", "Logistic Regression"]
train_accs = [dt_train_acc, lr_train_acc]
test_accs  = [dt_test_acc,  lr_test_acc]
f1_scores  = [dt_f1,        lr_f1]

x = np.arange(len(models))
w = 0.25
fig5, ax5 = plt.subplots(figsize=(9, 5))
ax5.bar(x - w, train_accs, w, label="Train Accuracy", color="#4e8cff", alpha=0.85)
ax5.bar(x,     test_accs,  w, label="Test Accuracy",  color="#06d6a0", alpha=0.85)
ax5.bar(x + w, f1_scores,  w, label="Macro F1",       color="#ffd166", alpha=0.85)
ax5.set_xticks(x)
ax5.set_xticklabels(models, fontsize=11)
ax5.set_ylim(0, 1.05)
ax5.set_ylabel("Score")
ax5.set_title("Model Comparison — Decision Tree vs Logistic Regression",
              fontsize=12, fontweight="bold")
ax5.legend(fontsize=9)
ax5.axhline(0.75, color="#ff6b6b", linestyle="--", linewidth=1, alpha=0.6, label="0.75 baseline")
for rect in ax5.patches:
    ax5.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005,
             f"{rect.get_height():.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig5.savefig(os.path.join("outputs", "5_model_comparison.png"),
             dpi=150, bbox_inches="tight", facecolor="#0f1117")
print("    Saved: 5_model_comparison.png")

print("\n" + "=" * 65)
print("  All outputs saved to outputs/")
print("=" * 65)
