# Day 5: Git, Collaboration & MLOps Foundations

> **Phase**: 1 - Foundations
> **Week**: 1 - The ML Engineer's Toolkit
> **Focus**: Version Control & Collaboration
> **Reading Time**: 50 mins

---

## 1. Git for Machine Learning

Git is the standard for version control in software engineering, but ML introduces a unique challenge: **The duality of Code and Data.**

### 1.1 The Binary File Problem
Git tracks line-based text changes. It computes diffs to show you what changed.
*   **Problem**: ML artifacts (images, datasets, trained model weights `.pth`, `.h5`) are binary files. A 1GB model file changes completely even if you change one weight. Git tries to store every version, bloating the repository size to gigabytes or terabytes, making `git clone` impossible.
*   **Rule**: **NEVER** commit large datasets or model weights to a standard Git repository.

### 1.2 Data Version Control (DVC)
DVC is "Git for Data."
*   **Mechanism**:
    1.  You add a large file `data.csv` to DVC.
    2.  DVC computes the hash of the file and stores it in a small text file `data.csv.dvc`.
    3.  You commit `data.csv.dvc` to Git.
    4.  You push the actual `data.csv` to a remote storage (S3, GCS, Azure Blob).
*   **Benefit**: This links a specific version of your code (Git commit) to a specific version of your data (DVC file). You can checkout any point in history and reproduce the exact state of code + data.

---

## 2. Collaboration Workflows

### 2.1 Branching Strategies
*   **Trunk-Based Development**: Developers merge small, frequent updates to the main branch. This is preferred for modern CI/CD and MLOps because it avoids "merge hell" and encourages continuous integration.
*   **Gitflow**: Uses long-lived feature branches and release branches. Often considered too heavy and slow for fast-paced ML experimentation.

### 2.2 Code Review in ML
Reviewing ML code is harder than standard software.
*   **Logic**: Is the algorithm correct?
*   **Math**: Is the loss function implemented correctly?
*   **Experiment Design**: Is there data leakage? (e.g., normalizing the whole dataset before splitting).
*   **Config**: Are the hyperparameters sensible?

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Jupyter Notebooks in Git
**Scenario**: You change one cell in a notebook. The diff shows 500 lines of changes because the execution count and metadata changed. Merging becomes a nightmare.
**Solution**:
1.  **Jupytext**: Automatically syncs notebooks to a clean `.py` file. You commit the `.py` file, which diffs beautifully.
2.  **nbstripout**: A git hook that strips output and metadata from notebooks before committing.

### Challenge 2: The Reproducibility Crisis
**Scenario**: "I ran the code and got 90% accuracy. You ran it and got 88%."
**Causes**:
- Random seed differences.
- Nondeterministic GPU operations (cuDNN).
- Different library versions.
**Solution**:
- **Seed Everything**: Set seeds for Python `random`, `numpy`, `torch`, and `cuda`.
- **Docker**: Enforce environment consistency.
- **Deterministic Flags**: Set `torch.backends.cudnn.deterministic = True` (might slow down training).

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: How do you handle large model weights in a repository?**
> **Answer**: Do not commit them to Git. Use **Git LFS (Large File Storage)** or, preferably, an MLOps artifact store like **MLflow**, **Weights & Biases**, or **DVC**. In the code, reference the artifact ID/URL. This keeps the repo lightweight and provides an audit trail of model versions.

**Q2: What is "Data Leakage" and how can code review catch it?**
> **Answer**: Data leakage occurs when information from the test/validation set influences the training process. Common examples:
> *   Normalizing/Scaling the entire dataset *before* splitting into train/test.
> *   Using future information (e.g., using "time of churn" feature to predict "will churn").
> *   Code review should specifically check the order of operations: Split FIRST, then Transform.

**Q3: Why is Trunk-Based Development preferred for MLOps?**
> **Answer**: ML experiments are often interdependent. Long-lived branches lead to massive divergence. Trunk-based development forces frequent integration, ensuring that changes to the data pipeline or model architecture are immediately visible to the whole team, reducing integration conflicts and accelerating the feedback loop.

### System Design Challenge
**Scenario**: Design a versioning system for a team of 10 ML engineers working on a shared dataset that updates weekly.
**Approach**:
1.  **Data Storage**: S3 bucket as the source of truth.
2.  **Versioning**: Use DVC. When the dataset updates, the data engineer runs `dvc add data/`, commits the `.dvc` file to Git, and pushes data to S3.
3.  **Access**: Engineers `git pull` the new `.dvc` file and run `dvc pull` to download only the new data changes.
4.  **Immutable Data**: S3 bucket should be versioned or read-only for engineers to prevent accidental overwrites.

---

## 5. Further Reading
- [DVC Documentation](https://dvc.org/)
- [Gitflow vs Trunk-Based Development](https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development)
- [Reproducibility in PyTorch](https://pytorch.org/docs/stable/notes/randomness.html)
