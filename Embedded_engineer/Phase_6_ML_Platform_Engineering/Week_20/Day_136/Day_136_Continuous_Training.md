# Day 136: The Loop: Continuous Training (CT)
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 20: CI/CD for ML

---

> **🎯 Focus Area:** Static models rot. The world changes (Concept Drift). **Continuous Training** ensures your model adapts automatically, triggered by schedule or data freshness, without human intervention.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** a Cron Schedule for model retraining (e.g., Weekly).
2.  **Architect** a Trigger-based system (New Data -> S3 Notification -> Lambda -> GitHub Dispatch -> Training).
3.  **Implement** a "Warm Start" strategy to cheapen retraining.
4.  **Handle** Training Failures gracefully (Alerts, Fallback to previous model).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install boto3 requests`.
- GitHub Repo.

---

## 📖 Theoretical Foundation

### 1. Schedule vs Trigger
*   **Schedule (Cron):** "Retrain every Sunday at 2 AM". Simple. Predictable cost. Good for seasonality.
*   **Trigger (Event-Driven):** "Retrain when 10k new samples arrive" or "Retrain when Accuracy < 80%". Reactive. Harder to budget.

### 2. The CT Pipeline Levels (Google MLOps Guide)
*   **Level 0:** Manual Training.
*   **Level 1:** ML Pipeline Automation (CT).
    *   Triggers.
    *   Data Validation.
    *   Model Validation.
    *   **The Goal:** The data scientist pushes code *once*, and the pipe runs *forever*.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: The Cron Trigger (GitHub Actions)

The simplest CT system is just a few lines of YAML.

#### 📁 `.github/workflows/scheduled-retraining.yaml`
```yaml
name: Weekly Retraining
on:
  schedule:
    - cron: '0 2 * * 0' # Every Sunday at 2 AM
  workflow_dispatch: # Allow manual run button

jobs:
  retrain:
    runs-on: ubuntu-latest
    permissions:
      contents: write # To commit metrics/model

    steps:
      - uses: actions/checkout@v3
      
      # ... Setup Python / CML ...
      
      - name: Retrain
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          # Pull LATEST data (Dynamic)
          dvc pull data/latest.csv
          
          # Train
          python src/run_pipeline.py run --build_id=weekly-${{ github.run_number }}
          
          # Verify Quality
          python src/check_quality.py --threshold 0.85

      - name: Register Model
        if: success()
        run: python src/register_model.py
```

### 👨‍💻 Infrastructure: Event-Driven Trigger (AWS Lambda -> GitHub)

Scenario: S3 `new_data/` receives file -> Automation starts.

#### 📁 `infra/lambda_function.py`
```python
import json
import os
import requests

GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
REPO_OWNER = "myuser"
REPO_NAME = "myrepo"
WORKFLOW_ID = "event-retraining.yaml"

def lambda_handler(event, context):
    print("Received S3 Event")
    
    # 1. Trigger GitHub Workflow Dispatch
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_ID}/dispatches"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {"ref": "main"}
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 204:
        print("Successfully triggered training.")
        return {"statusCode": 200, "body": "Usage Triggered"}
    else:
        print(f"Failed: {response.text}")
        return {"statusCode": 500, "body": "Failed"}
```

### 👨‍💻 Core Implementation: Warm Start

Retraining from scratch (Random Weights) is wasteful if data drift involves only 5% new data.

```python
# src/pipeline_components.py

def train(..., warm_start_model_uri=None):
    model = Model()
    
    if warm_start_model_uri:
        print(f"Warm starting from {warm_start_model_uri}")
        # Download old weights
        prev_model = mlflow.sklearn.load_model(warm_start_model_uri)
        # Transfer weights (Generic logic specific to library)
        model.load_state_dict(prev_model.state_dict())
    
    # Train for fewer epochs because we started close to optima
    model.fit(X, y, epochs=5)
```

---

## 🔬 Lab Exercise: "Circuit Breaker"

### Task
Prevent bad loops.
1.  Setup: Trigger retraining on "Accuracy Drop".
2.  Scenario: Drastic Concept Drift (COVID-19 happens).
3.  Effect: Model accuracy < Threshold. -> Retrain.
4.  New Model accuracy < Threshold. -> Retrain.
5.  **Infinite Loop!**
6.  **Fix:** Implement a "Cooldown" or "Max Retries". Check DB: "If last retraining was < 24h ago, do not trigger."

---

## 📝 Daily Summary

### Key Takeaways
1.  **Freshness:** The value of an ML model degrades over time (Age of Information). CT maintains value.
2.  **Cost:** Frequent retraining is expensive. Balance "Freshness" vs "Compute Bill".
3.  **Monitoring:** You need a dashboard showing "Last Successful Training Time". If it's > 1 month, alarm should ring.

### API Summary
```yaml
on:
  schedule:
    - cron: '...'
```

---

**Day 136 Complete** ✅

*Next: Day 137 - Model Testing - Unit, Integration, and Data Tests.*
