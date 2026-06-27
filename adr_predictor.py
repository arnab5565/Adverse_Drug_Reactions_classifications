"""
ADR Predictor - Simple Wrapper for Easy Testing
Just import and use - no need to worry about preprocessing!
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class ADRPredictor:
    """
    Simple wrapper for testing ADR predictions
    
    Usage:
        predictor = ADRPredictor()
        result = predictor.predict(age_group="41-60", gender="Female", ...)
    """
    
    SEVERITY_LABELS = ["No Reaction", "Hospitalization", "Life-Threatening/Disabling", "Death"]
    
    def __init__(self):
        """Initialize and train models"""
        self.dt_model = None
        self.lr_model = None
        self.le_dict = {}
        self._train_models()
    
    def _train_models(self):
        """Train models on synthetic data"""
        print("Training ADR models...")
        
        # Generate synthetic training data
        np.random.seed(42)
        n = 2000
        
        age_bins = ["0-20", "21-40", "41-60", "61-80", "80+"]
        df = pd.DataFrame({
            "age_group": np.random.choice(age_bins, n, p=[0.06, 0.18, 0.32, 0.30, 0.14]),
            "gender": np.random.choice(["Female", "Male"], n, p=[0.56, 0.44]),
            "dosage_form": np.random.choice(["Tablet", "Injection", "Capsule", "Solution", "Patch", "Inhaler"], n, p=[0.35, 0.25, 0.18, 0.10, 0.07, 0.05]),
            "drug_class": np.random.choice(["Antibiotic", "Anticoagulant", "NSAID", "Antihypertensive", "Chemotherapy", "Antidiabetic", "Antidepressant"], n),
            "country": np.random.choice(["US", "CA", "UK", "DE", "Other"], n, p=[0.55, 0.27, 0.08, 0.05, 0.05]),
            "num_drugs": np.random.randint(1, 8, n),
            "prior_adr": np.random.randint(0, 2, n),
            "renal_impairment": np.random.choice([0, 1], n, p=[0.75, 0.25]),
            "hepatic_impairment": np.random.choice([0, 1], n, p=[0.80, 0.20]),
        })
        
        # Create severity scores
        score = np.zeros(n)
        score += np.where(df["dosage_form"] == "Injection", 1.2, 0)
        score += np.where(df["dosage_form"] == "Tablet", 0.5, 0)
        score += np.where(df["drug_class"] == "Chemotherapy", 1.5, 0)
        score += np.where(df["drug_class"] == "Anticoagulant", 1.0, 0)
        score += np.where(df["drug_class"] == "Antibiotic", 0.4, 0)
        score += np.where(df["age_group"] == "80+", 1.2, 0)
        score += np.where(df["age_group"] == "61-80", 0.7, 0)
        score += np.where(df["age_group"] == "0-20", 0.3, 0)
        score += df["num_drugs"] * 0.15
        score += df["prior_adr"] * 0.8
        score += df["renal_impairment"] * 0.9
        score += df["hepatic_impairment"] * 0.7
        score += np.random.normal(0, 0.5, n)
        
        thresholds = np.percentile(score, [50, 72, 88])
        df["severity"] = np.digitize(score, thresholds)
        
        # Preprocess
        df_enc = df.copy()
        cat_cols = ["age_group", "gender", "dosage_form", "drug_class", "country"]
        for col in cat_cols:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df[col])
            self.le_dict[col] = le
        
        X = df_enc.drop("severity", axis=1)
        y = df_enc["severity"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Decision Tree
        self.dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
        self.dt_model.fit(X_train, y_train)
        
        # Train Logistic Regression
        self.lr_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        ])
        self.lr_model.fit(X_train, y_train)
        
        print(f"✓ Models trained!")
        print(f"  Decision Tree accuracy: {self.dt_model.score(X_test, y_test):.2%}")
        print(f"  Logistic Regression accuracy: {self.lr_model.score(X_test, y_test):.2%}\n")
    
    def predict(self, age_group, gender, dosage_form, drug_class, country="US",
                num_drugs=1, prior_adr=0, renal_impairment=0, hepatic_impairment=0,
                model="both", return_confidence=True):
        """
        Predict ADR severity for a patient
        
        Args:
            age_group: "0-20", "21-40", "41-60", "61-80", or "80+"
            gender: "Female" or "Male"
            dosage_form: "Tablet", "Injection", "Capsule", "Solution", "Patch", "Inhaler"
            drug_class: "Antibiotic", "Anticoagulant", "NSAID", etc.
            country: "US", "CA", "UK", "DE", or "Other" (default: "US")
            num_drugs: 1-8 (default: 1)
            prior_adr: 0 or 1 (default: 0)
            renal_impairment: 0 or 1 (default: 0)
            hepatic_impairment: 0 or 1 (default: 0)
            model: "dt" for Decision Tree, "lr" for Logistic Regression, "both" (default)
            return_confidence: Include confidence scores (default: True)
        
        Returns:
            dict with predictions and confidence scores
        """
        # Create patient data
        patient = pd.DataFrame({
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
        
        # Preprocess
        patient_enc = patient.copy()
        for col in self.le_dict:
            try:
                patient_enc[col] = self.le_dict[col].transform(patient[col])
            except ValueError:
                patient_enc[col] = 0  # Default to first class
        
        X = patient_enc.values
        
        result = {
            "patient": patient.iloc[0].to_dict(),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        if model in ["dt", "both"]:
            dt_pred = self.dt_model.predict(X)[0]
            result["decision_tree"] = {
                "severity_class": dt_pred,
                "severity_label": self.SEVERITY_LABELS[dt_pred]
            }
            if return_confidence and hasattr(self.dt_model, 'predict_proba'):
                conf = self.dt_model.predict_proba(X).max()
                result["decision_tree"]["confidence"] = float(conf)
        
        if model in ["lr", "both"]:
            lr_pred = self.lr_model.predict(X)[0]
            result["logistic_regression"] = {
                "severity_class": lr_pred,
                "severity_label": self.SEVERITY_LABELS[lr_pred]
            }
            if return_confidence and hasattr(self.lr_model, 'predict_proba'):
                conf = self.lr_model.predict_proba(X).max()
                result["logistic_regression"]["confidence"] = float(conf)
        
        return result
    
    def predict_batch(self, df, model="both"):
        """
        Predict on multiple patients from DataFrame
        
        Args:
            df: DataFrame with columns [age_group, gender, dosage_form, drug_class, ...]
            model: "dt", "lr", or "both"
        
        Returns:
            DataFrame with predictions
        """
        predictions = []
        for idx, row in df.iterrows():
            result = self.predict(**row.to_dict(), model=model)
            predictions.append(result)
        
        return predictions
    
    def print_result(self, result, verbose=True):
        """Pretty print prediction result"""
        print("\n" + "="*70)
        print("  ADR SEVERITY PREDICTION")
        print("="*70)
        
        if verbose:
            print("\nPATIENT INFO:")
            for k, v in result["patient"].items():
                print(f"  {k:.<30} {v}")
        
        if "decision_tree" in result:
            dt = result["decision_tree"]
            print(f"\nDECISION TREE:")
            print(f"  Severity ..................... {dt['severity_label']} (class {dt['severity_class']})")
            if "confidence" in dt:
                print(f"  Confidence ................... {dt['confidence']:.1%}")
        
        if "logistic_regression" in result:
            lr = result["logistic_regression"]
            print(f"\nLOGISTIC REGRESSION:")
            print(f"  Severity ..................... {lr['severity_label']} (class {lr['severity_class']})")
            if "confidence" in lr:
                print(f"  Confidence ................... {lr['confidence']:.1%}")
        
        # Consensus check
        if "decision_tree" in result and "logistic_regression" in result:
            dt_class = result["decision_tree"]["severity_class"]
            lr_class = result["logistic_regression"]["severity_class"]
            if dt_class == lr_class:
                print(f"\n✓ CONSENSUS: Both models agree")
            else:
                print(f"\n⚠ DISAGREEMENT: Models predict different classes")
        
        print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Initialize predictor
    predictor = ADRPredictor()
    
    # Example 1: Single prediction
    print("\n[EXAMPLE 1] Single Patient Prediction")
    result = predictor.predict(
        age_group="41-60",
        gender="Female",
        dosage_form="Injection",
        drug_class="Anticoagulant",
        num_drugs=3,
        prior_adr=1,
        renal_impairment=0,
        hepatic_impairment=1
    )
    predictor.print_result(result)
    
    # Example 2: Quick prediction (minimal code)
    print("\n[EXAMPLE 2] Quick Prediction")
    result2 = predictor.predict("21-40", "Male", "Tablet", "Antibiotic")
    predictor.print_result(result2, verbose=False)
    
    # Example 3: Batch predictions
    print("\n[EXAMPLE 3] Batch Predictions")
    batch_df = pd.DataFrame({
        "age_group": ["80+", "41-60", "21-40"],
        "gender": ["Female", "Male", "Female"],
        "dosage_form": ["Patch", "Injection", "Tablet"],
        "drug_class": ["NSAID", "Chemotherapy", "Antibiotic"],
    })
    
    batch_results = predictor.predict_batch(batch_df)
    for i, result in enumerate(batch_results, 1):
        print(f"\nPatient {i}:")
        print(f"  DT: {result['decision_tree']['severity_label']}")
        print(f"  LR: {result['logistic_regression']['severity_label']}")
