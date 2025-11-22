# Lab 3.10: Version Control Capstone Project

## 🎯 Objective

Simulate a complete team workflow. You will act as two developers ("Dev A" and "Dev B") working on a shared repository, using Feature Branches, Pull Requests, Code Reviews, and resolving Merge Conflicts.

## 📋 Prerequisites

-   Completed Module 3.
-   GitHub Account.

## 📚 Background

### The Scenario

You are building a **Calculator App**.
-   **Dev A (You)**: Implementing Addition.
-   **Dev B (Also You)**: Implementing Subtraction.
-   **Constraint**: Both modify `calculator.py`.

---

## 🔨 Hands-On Implementation

### Step 1: Setup (The Repo) 🏗️

1.  Create a new GitHub Repo `calculator-app`.
2.  Clone it locally.
3.  Create `calculator.py`:
    ```python
    class Calculator:
        def start(self):
            print("Calculator started")
    ```
4.  Commit and Push to `main`.

### Step 2: Dev A (Addition) ➕

1.  Create branch `feature/add`.
2.  Modify `calculator.py`:
    ```python
    class Calculator:
        def add(self, a, b):
            return a + b
            
        def start(self):
            print("Calculator started")
    ```
3.  Commit: "Feat: Add addition function".
4.  Push: `git push -u origin feature/add`.
5.  **GitHub**: Create Pull Request. Merge it.

### Step 3: Dev B (Subtraction) ➖

*Simulate a second developer who hasn't pulled yet.*

1.  Switch back to `main` (Locally).
2.  **DO NOT PULL** (We want to create a conflict).
3.  Create branch `feature/sub`.
4.  Modify `calculator.py` (Dev B doesn't know about `add` yet):
    ```python
    class Calculator:
        def sub(self, a, b):
            return a - b

        def start(self):
            print("Calculator started")
    ```
    *Note:* Place `sub` exactly where `add` was placed by Dev A (top of class).
5.  Commit: "Feat: Add subtraction".
6.  Push: `git push -u origin feature/sub`.

### Step 4: The Conflict 💥

1.  **GitHub**: Create Pull Request for `feature/sub`.
2.  **GitHub**: It will say "Can't automatically merge".
3.  **Local Resolution**:
    ```bash
    git checkout feature/sub
    git pull origin main  # Pulls Dev A's changes into Dev B's branch
    ```
    *Result:* Conflict in `calculator.py`.

4.  **Fix it**:
    Keep BOTH functions.
    ```python
    class Calculator:
        def add(self, a, b):
            return a + b

        def sub(self, a, b):
            return a - b
            
        def start(self):
            print("Calculator started")
    ```
5.  **Commit & Push**:
    ```bash
    git add calculator.py
    git commit -m "Merge main and fix conflict"
    git push
    ```

6.  **GitHub**: Merge the PR.

### Step 5: Tagging Release 🏷️

1.  Switch to `main`.
2.  Pull everything.
    ```bash
    git checkout main
    git pull
    ```
3.  Verify both functions exist.
4.  Tag v1.0.
    ```bash
    git tag -a v1.0 -m "Release 1.0"
    git push origin v1.0
    ```

---

## 🎯 Challenges

### Challenge 1: Branch Protection (Difficulty: ⭐⭐⭐)

**Task:**
Go to GitHub Repo Settings > Branches.
Add a rule for `main`:
-   Require Pull Request reviews before merging.
-   Try to push directly to `main` from your terminal. It should fail.

### Challenge 2: Revert the Release (Difficulty: ⭐⭐⭐)

**Scenario:** The subtraction function has a bug (it returns `a + b` by mistake).
**Task:**
1.  Revert the merge commit of `feature/sub`.
2.  Push the revert.
3.  Verify `sub` is gone but `add` remains.

---

## 🔑 Key Takeaways

1.  **Sync Early, Sync Often**: If Dev B had pulled `main` before starting, the conflict might have been avoided or easier.
2.  **PRs are Safety Nets**: They run tests (if configured) and allow review.
3.  **Tags are Permanent**: Branches move, tags stay. Use them for releases.

---

## ⏭️ Next Steps

**Congratulations!** You have completed Module 3: Version Control.
You are now ready to build networks.

Proceed to **Module 4: Networking Fundamentals**.
