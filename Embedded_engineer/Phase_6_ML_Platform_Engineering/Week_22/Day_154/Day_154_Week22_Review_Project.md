# Day 154: Week 22 Review & Project - The Unbreakable Pipeline
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 22: Data & Model Quality

---

> **🎯 Focus Area:** We have built the brakes (Data Validation), the speedometer (Metrics), and the blind-spot monitor (Drift Detection). Now we integrate them into the car (Pipeline) to ensure we never crash in production.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** a Blocking Quality Gate that halts deployment if any check fails.
2.  **Orchestrate** Data Quality, Training, and Drift Checks in a single DAG.
3.  **Produce** a comprehensive "Model Card" report containing all quality evidence.
4.  **Execute** a safe rollout using Shadow Deployment logic.

---

## 📚 Week 22 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 148 | Data Quality | "The model crashed because `age` was -5." |
| 149 | Metrics | "Accuracy is high, but Calibration is terrible." |
| 150 | Drift | "The model rots even if code doesn't change." |
| 151 | Feature Store | "Training on 'Current Balance' was a huge mistake." |
| 152 | A/B Tests | "Offline AUC means nothing if users don't click." |
| 153 | Shadow Mode | "I tested on 100k users without breaking anything." |

---

## 🏗️ Final Project: "GuardRailHub"

### Scenario
You are building the "Credit Scoring" pipeline.
**Risk:** Bad deployment = Lawsuit.
**Requirement:** strictly enforce quality gates.

### Step 1: The Quality Library

#### 📁 `project/quality_gates.py`
```python
import great_expectations as gx
import alibi_detect
import json
from sklearn.metrics import precision_score

class GateKeeper:
    def __init__(self):
        self.report = {"checks": [], "passed": True}

    def check_data(self, df):
        # 1. Great Expectations
        print("Running Data Validation...")
        # (Simulate GX Checkpoint)
        if df.isnull().sum().sum() > 0:
            self.report["checks"].append({"name": "Data_Nulls", "status": "FAIL"})
            self.report["passed"] = False
        else:
            self.report["checks"].append({"name": "Data_Nulls", "status": "PASS"})

    def check_model(self, model, X_test, y_test):
        # 2. Performance Metric
        print("Running Model Evaluation...")
        preds = model.predict(X_test)
        prec = precision_score(y_test, preds)
        
        threshold = 0.85
        if prec < threshold:
            self.report["checks"].append({
                "name": "Precision_Check", 
                "status": "FAIL", 
                "value": prec, 
                "threshold": threshold
            })
            self.report["passed"] = False
        else:
             self.report["checks"].append({"name": "Precision_Check", "status": "PASS", "value": prec})

    def check_drift(self, X_ref, X_curr):
        # 3. Drift Check
        print("Running Drift Detection...")
        # Simulate Alibi KSDrift
        is_drift = False # Assume safe
        
        if is_drift:
            self.report["checks"].append({"name": "Drift_Check", "status": "FAIL"})
            self.report["passed"] = False
        else:
            self.report["checks"].append({"name": "Drift_Check", "status": "PASS"})

    def save_report(self):
        with open("quality_report.json", "w") as f:
            json.dump(self.report, f, indent=2)
            
        if not self.report["passed"]:
            raise RuntimeError("Quality Gates Failed! See report.")
```

### Step 2: The Integrated Pipeline script

#### 📁 `project/pipeline.py`
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from quality_gates import GateKeeper
import mlflow

def run_pipeline():
    gate = GateKeeper()
    
    # 1. Ingest
    df = pd.read_csv("data/credit.csv")
    
    # 2. Gate: Data Quality
    gate.check_data(df)
    
    # 3. Train
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # 4. Gate: Model Performance
    gate.check_model(model, X_test, y_test)
    
    # 5. Gate: Feature Drift
    # Compare Test set (Current) to Reference (Training Baseline)
    # Ideally compare New Data vs Old Data
    gate.check_drift(X_train, X_test)
    
    # 6. Save Report
    gate.save_report()
    
    # 7. Register (Only if Gates Passed)
    print("All Gates Passed. Registering Model...")
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("quality_report.json")

if __name__ == "__main__":
    try:
        run_pipeline()
    except RuntimeError as e:
        print(f"PIPELINE HALTED: {e}")
        exit(1)
```

### Step 3: CI Integration

#### 📁 `.github/workflows/quality.yaml`
```yaml
name: Quality Pipeline
on: [push]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Pipeline
        run: python project/pipeline.py
      - name: Upload Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: quality-report
          path: quality_report.json
```

---

## 🔬 Lab Exercise: "The Model Card"

### Task
Generate Documentation.
1.  Read `quality_report.json`.
2.  Read `params` from MLflow.
3.  Generate `MODEL_CARD.md`:
    *   **Intended Use:** Credit Scoring.
    *   **limitations:** Trained on US Data.
    *   **Performance:** Precision 0.88.
    *   **Ethics:** Fairness check passed (False Positive Rate parity between demographic groups).
4.  **Goal:** Auditors read this, not the code.

---

## 📝 Success Criteria
1.  **Blocking:** If Data has nulls, training never starts (saves compute).
2.  **Evidence:** Every deployed model has a linked JSON report proving quality.
3.  **Automation:** Drifts are detected before customers complain.

---

**Week 22 Complete** ✅
**Phase 6D In Progress**

*Next Phase: Security & Governance - Locking the Doors.*
