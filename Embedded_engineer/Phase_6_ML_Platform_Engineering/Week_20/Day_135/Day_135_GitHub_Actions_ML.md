# Day 135: The Robot Colleague: GitHub Actions for ML
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 20: CI/CD for ML

---

> **🎯 Focus Area:** CI isn't just for running `pytest`. It's for training models. When you open a Pull Request, a bot should fetch your data, retrain the model, and comment on your PR with the new Accuracy vs the Main Branch.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** a Self-Hosted Runner on a GPU machine (EC2/GCP).
2.  **Author** a GitHub Actions Workflow (`.github/workflows/ml-ci.yml`).
3.  **Use** CML (Continuous Machine Learning) to post markdown reports to PRs.
4.  **Manage** Secrets (AWS Keys, W&B Keys) securely in the CI environment.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- A GitHub Repository.
- A Cloud VM (Optional for Self-Hosted) or Standard GitHub Runners (CPU only).

### Software Environment
- `pip install cml dvc`.

---

## 📖 Theoretical Foundation

### 1. CI/CD for Code vs Data
*   **Web App CI:** Build -> Test -> Deploy. Takes 5 mins.
*   **ML CI:** Data Ingest -> Feature Eng -> Train (Hours) -> Eval.
*   **Challenge:** GitHub-hosted runners are slow (2 vCPUs) and have no GPUs.
*   **Solution:** **Self-Hosted Runners** or **Cloud Dispatch** (Trigger SageMaker/Vertex from GHA).

### 2. CML (Continuous Machine Learning)
A tool by Iterative.ai. It helps cloud runners generate reports (Plots, Tables) and push them back to GitHub as PR comments.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Self-Hosted Runner (Setup)

If we want to train small models inside CI, we need a GPU runner.
Run this on your AWS `p3.2xlarge`:

```bash
# 1. Download Runner
curl -o actions-runner-linux-x64-2.304.0.tar.gz -L https://github.com/actions/runner/releases/download/...
tar xzf ./actions-runner-linux-x64-2.304.0.tar.gz

# 2. Configure (Token from GitHub Repo -> Settings -> Actions -> Runners)
./config.sh --url https://github.com/myuser/myrepo --token ABCdef123

# 3. Run
./run.sh
# or install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

### 👨‍💻 Core Implementation: The Workflow

#### 📁 `.github/workflows/train-on-pr.yaml`
```yaml
name: Model Training CI
on: [pull_request]

jobs:
  train-and-report:
    runs-on: ubuntu-latest # Or [self-hosted, gpu]
    container: docker://dvcorg/cml-py3:latest # Has CML pre-installed

    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install Dependencies
        run: pip install -r requirements.txt

      - name: Pull Data (DVC)
        env:
          GDRIVE_CREDENTIALS_DATA: ${{ secrets.GDRIVE_CREDENTIALS_DATA }}
        run: dvc pull data/raw.csv

      - name: Train Model
        run: |
          # Run the pipeline defined in Day 134
          python src/run_pipeline.py run --build_id=${{ github.sha }}

      - name: Write CML Report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Create Markdown Report
          echo "## Model Metrics for commit ${{ github.sha }}" > report.md
          cat artifacts/${{ github.sha }}/metrics.json >> report.md
          
          echo "## Confusion Matrix" >> report.md
          # Assume train generated confusion_matrix.png
          cml-publish confusion_matrix.png --md >> report.md
          
          # Send to PR
          cml-send-comment report.md
```

### 👨‍💻 Core Implementation: Training Script Adaptation

Your script needs to produce the artifacts `metrics.json` and `confusion_matrix.png` expected by the yaml.

```python
# In src/pipeline_components.py (evaluate step)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate(...):
    # ... logic ...
    
    # Save Image
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig("confusion_matrix.png")
```

---

## 🔬 Lab Exercise: "Resource Limits"

### Task
Trigger a timeout.
1.  Use standard GitHub Runner.
2.  Pipeline trains for 1 hour.
3.  **Observation:** GitHub creates a bill (per minute billing). Or times out (6h limit).
4.  **Fix:** CI should verify *Code Correctness* on a *Subset* of data (Fast).
    *   `if CI == True: data = data.sample(frac=0.01)`
    *   Full training happens only on Merge to Main, or via explicit `/train` comment command.

---

## 📖 Advanced Theory: ChatOps
Instead of running on every push, use **ChatOps**.
1.  Developer comments on PR: `/train-full-gpu`
2.  GHA trigger: `on: issue_comment`
3.  Workflow parses comment.
4.  Provisions Spot Instance. Runs Training. Comments back results. Terminates instance.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Credentials:** Never commit AWS Keys. Use Repo Secrets. Inject them as Environment Variables.
2.  **CML:** The ability to see the Loss Curve *in the Pull Request* prevents merging regressions.
3.  **Hygiene:** Run `black` (Linting) and `pytest` (Unit Tests) before attempting Training. Fail fast.

### API Summary
```yaml
uses: actions/checkout@v3
run: python train.py
```

---

**Day 135 Complete** ✅

*Next: Day 136 - Continuous Training (CT) - Automation Loop.*
