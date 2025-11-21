# Day 5: Git & MLOps - Interview Questions

> **Topic**: Version Control & Collaboration
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the difference between `git merge` and `git rebase`?
**Answer:**
*   **Merge**: Creates a new "merge commit" tying two histories together. Preserves the exact history of when changes happened. Non-destructive.
*   **Rebase**: Moves the entire feature branch to begin on the tip of the main branch. Rewrites history. Creates a linear history (cleaner) but can be dangerous on shared branches.

### 2. Explain the concept of a "Commit Hash" (SHA-1). How is it generated?
**Answer:**
*   It is a checksum of the **content** (files), the **metadata** (author, time), and the **parent hash**.
*   This makes Git a Merkle Tree. If you change one bit in history, all subsequent hashes change.

### 3. What is `git stash`? When would you use it?
**Answer:**
*   Temporarily shelves (saves) changes that are not ready to be committed so you can switch branches.
*   **Use Case**: You are working on Feature A, but a critical bug comes in. `git stash`, switch to hotfix branch, fix bug, switch back, `git stash pop`.

### 4. How do you resolve a Merge Conflict?
**Answer:**
*   Git marks the file with `<<<<<<<`, `=======`, `>>>>>>>`.
*   You manually edit the file to choose the correct code (or combine both).
*   `git add` the resolved file and `git commit`.

### 5. What is the difference between `git fetch` and `git pull`?
**Answer:**
*   `git fetch`: Downloads changes from remote to your local `.git` folder but **does not** modify your working files. Safe.
*   `git pull`: `git fetch` + `git merge`. Downloads and immediately tries to merge into your current code. Can cause conflicts.

### 6. Why shouldn't you commit large files (like models or datasets) to Git?
**Answer:**
*   Git is designed for text. It stores every version of every file.
*   Committing a 1GB model makes the repository huge (bloat). Cloning becomes slow.
*   Git struggles with diffing binary files.

### 7. What is DVC (Data Version Control)? How does it differ from Git LFS?
**Answer:**
*   **Git LFS**: Extension to Git. Replaces large files with pointers. Tightly coupled with Git server.
*   **DVC**: Cloud-agnostic. Stores data in S3/GCS/Azure. Stores small `.dvc` pointer files in Git. Also manages **pipelines** and **experiments**, not just file storage.

### 8. Explain the command `dvc repro`. How does it know what to run?
**Answer:**
*   `dvc repro` reproduces the pipeline defined in `dvc.yaml`.
*   It checks the **dependencies** (input files/hashes) and **outputs**.
*   If dependencies haven't changed (hashes match `dvc.lock`), it skips the stage (caching). If they changed, it re-runs the command.

### 9. What is a DAG (Directed Acyclic Graph) in the context of MLOps pipelines?
**Answer:**
*   A graph where nodes are tasks (Preprocessing -> Training -> Eval) and edges are dependencies.
*   **Acyclic**: No loops.
*   Tools like Airflow or DVC build a DAG to determine execution order and parallelism.

### 10. How do you revert a bad commit in Git that has already been pushed to main?
**Answer:**
*   **Do NOT use reset** (it rewrites history and breaks it for others).
*   Use `git revert <commit_hash>`. This creates a *new* commit that is the exact inverse of the bad commit.

### 11. What is "Continuous Integration" (CI) for Machine Learning?
**Answer:**
*   Automated testing whenever code is pushed.
*   **ML Specifics**: Besides unit tests, CI for ML might run:
    *   Linting (Ruff).
    *   Data validation (Great Expectations).
    *   Small model training (smoke test) to ensure no crashes.

### 12. Explain the folder structure of a standard `.dvc` cache.
**Answer:**
*   It's a content-addressable storage (CAS).
*   Files are renamed to their MD5 hash.
*   First 2 chars of hash = Directory name. Remaining chars = Filename.
*   Example: `a3/f29...`.

### 13. What is a "Feature Branch" workflow?
**Answer:**
*   Main branch is protected.
*   Developers create a new branch `feature/new-model` for every task.
*   Open a Pull Request (PR) to merge back to main. Code review happens on the PR.

### 14. How do you tag a specific version of a model release in Git?
**Answer:**
*   `git tag -a v1.0 -m "Release model v1"`.
*   Since DVC links data to Git commits, tagging the Git commit effectively tags the model version associated with it.

### 15. What is `git cherry-pick`?
**Answer:**
*   Applying the changes introduced by a specific commit from one branch onto another branch.
*   Useful for porting a specific bug fix from `dev` to `prod` without merging the whole `dev` branch.

### 16. How does DVC handle data deduplication?
**Answer:**
*   Since files are stored by their hash, identical files (even with different names) have the same hash.
*   DVC stores only one copy in the cache.

### 17. What is the difference between a "Soft Reset", "Mixed Reset", and "Hard Reset"?
**Answer:**
*   **Soft**: Moves HEAD back. Changes are left in "Staged" area.
*   **Mixed** (Default): Moves HEAD back. Changes are left in "Working Directory" (Unstaged).
*   **Hard**: Moves HEAD back. **Destroys** all changes. Dangerous.

### 18. How do you ensure reproducibility in an ML experiment?
**Answer:**
*   **Code**: Git commit hash.
*   **Data**: DVC hash / S3 path.
*   **Env**: Docker image / Lock file.
*   **Randomness**: Set global random seeds (NumPy, PyTorch, Python).

### 19. What is a `.gitignore` file? Give examples of files that should be ignored in an ML project.
**Answer:**
*   Tells Git what to ignore.
*   **Examples**:
    *   `__pycache__/`
    *   `*.pyc`
    *   `.env` (Secrets!)
    *   `venv/`
    *   `data/` (Use DVC instead)
    *   `models/*.pt`

### 20. Explain the concept of "Experiment Tracking" (e.g., MLflow/W&B).
**Answer:**
*   Logging metadata (hyperparams, metrics, loss curves) for every run.
*   Unlike Git (which tracks code), these tools track the *results* of running the code.
*   Allows comparing Run A vs Run B to see which learning rate worked best.
