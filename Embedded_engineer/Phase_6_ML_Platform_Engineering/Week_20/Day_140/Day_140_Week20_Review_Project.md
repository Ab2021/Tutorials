# Day 140: Week 20 Review & Project - The Automated MLOps Factory
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 20: CI/CD for ML

---

> **🎯 Focus Area:** We have built the conveyor belt segments (Actions, Training, Testing, Deployment, IaC). Now, we bolt them together into a single, unified factory that turns "Code" into "Served Prediction" without human touch.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** CML, DVC, and MLflow into a single GitHub Actions workflow.
2.  **Gate** deployments based on "Golden Testing" (High confidence on reference set).
3.  **Provision** the entire platform (buckets, users, instances) using a `make infra` command wrapping Terraform.
4.  **Demonstrate** a full loop: `git push` -> Report -> Merge -> Train -> Deploy.

---

## 📚 Week 20 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 134 | Pipelines | "Monolithic scripts break. Modular DAGs survive." |
| 135 | Actions | "I can see the Confusion Matrix in the PR comments!" |
| 136 | CT | "The model retrains itself every Sunday." |
| 137 | Testing | "Unit tests catch bugs. Behavioral tests catch stupidity." |
| 138 | CD | "Canary deployment saved us from a 500 error spike." |
| 139 | IaC | "I destroyed the whole stack and rebuilt it in 5 minutes." |

---

## 🏗️ Final Project: "ChurnOps"

### Scenario
We are building a Churn Prediction Service.
**Requirements:**
1.  PRs must prove they don't crash.
2.  Merges to `main` trigger full training.
3.  Successful training triggers Staging deployment.
4.  Manual Approval triggers Production deployment.

### Step 1: The Repo Structure

```
churn-ops/
├── .github/workflows/
│   ├── pr-check.yaml
│   └── train-release.yaml
├── infra/                  # Terraform
├── src/                    # Pipeline Code
│   ├── train.py
│   └── test.py
├── data/                   # DVC pointers
└── requirements.txt
```

### Step 2: The PR Checker (Fast)

#### 📁 `.github/workflows/pr-check.yaml`
```yaml
name: PR Verification
on: [pull_request]

jobs:
  fast-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      
      # 1. Unit Tests
      - name: Install
        run: pip install -r requirements.txt
      - name: Pytest
        run: pytest tests/unit/
        
      # 2. Smoke Test (Train on 1% data)
      - name: Smoke Train
        run: |
          # Use DVC to pull small subset (configured via different remote or manually sampled)
          # For demo, just sample local file
          python src/train.py --smoke_test --epochs 1
          
      # 3. Report
      - name: CML Report
        env:
           REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
           echo "# Smoke Test Report" > report.md
           cat metrics.json >> report.md
           cml-send-comment report.md
```

### Step 3: The Release Pipeline (Slow)

#### 📁 `.github/workflows/train-release.yaml`
```yaml
name: Release Pipeline
on:
  push:
    branches: [main]

jobs:
  full-train:
    runs-on: [self-hosted, gpu] # Our Terraform-provisioned runner
    outputs:
      model_uri: ${{ steps.register.outputs.uri }}
    
    steps:
      - uses: actions/checkout@v3
      - name: Train & Register
        id: register
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          dvc pull data/full
          python src/train.py --full
          uri=$(python src/get_latest_model.py)
          echo "uri=$uri" >> $GITHUB_OUTPUT

  deploy-staging:
    needs: full-train
    runs-on: ubuntu-latest
    steps:
      - name: Deploy Staging
        run: |
          python scripts/deploy.py --env staging --uri ${{ needs.full-train.outputs.model_uri }}
          
      - name: Integration Test
        run: pytest tests/integration/ --url http://staging.churn.internal

  deploy-prod:
    needs: deploy-staging
    environment: production # Requires Manual Approval in settings
    runs-on: ubuntu-latest
    steps:
      - name: Canary Rollout
        run: |
          python scripts/rollout.py --uri ${{ needs.full-train.outputs.model_uri }}
```

### Step 4: The Terraform Infrastructure

#### 📁 `infra/main.tf`
```hcl
module "runner" {
  source = "./modules/gh_runner"
  repo   = "myorg/churn-ops"
  token  = var.gh_token
}

module "k8s_cluster" {
  source = "./modules/eks"
  node_groups = {
    cpu_pool = { instance_types = ["t3.medium"] } # Serving
    gpu_pool = { instance_types = ["g4dn.xlarge"] } # Training
  }
}
```

---

## 🔬 Lab Exercise: "The Bad PR"

### Task
Simulate a regression.
1.  Create a branch `bug/breaking-change`.
2.  Modify `src/train.py`: invert the loss function (`loss = -mse`).
3.  Push.
4.  **Expectation:**
    *   Unit tests passed? Only if you don't test loss direction.
    *   Smoke Test: Loss is negative infinity.
    *   CML Report: Shows weird metrics.
    *   **Action:** Reviewer sees the report and Rejects the PR. "Please fix loss function."

---

## 📝 Success Criteria
1.  **Automation:** No step requires SSH or manual CLI execution.
2.  **Visibility:** The team knows the status of the model via GitHub Comments and Actions Tab.
3.  **Safety:** Production is protected by Staging Tests and Manual Gates.

---

**Week 20 Complete** ✅
**Phase 6D In Progress**

*Next Phase: Observability - Seeing inside the Black Box.*
