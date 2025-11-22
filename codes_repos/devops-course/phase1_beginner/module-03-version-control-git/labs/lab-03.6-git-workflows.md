# Lab 3.6: Git Workflows (GitFlow vs Trunk-Based)

## 🎯 Objective

Understand the two dominant strategies for managing Git in teams. You will simulate a "GitFlow" release cycle and a "Trunk-Based" rapid development cycle.

## 📋 Prerequisites

-   Completed Lab 3.5.

## 📚 Background

### 1. GitFlow (Traditional)
Designed for scheduled releases (e.g., "Version 2.0").
-   **Branches**: `main`, `develop`, `feature/*`, `release/*`, `hotfix/*`.
-   **Pros**: Strict control, stable main.
-   **Cons**: Complex, slow integration (Merge Hell).

### 2. Trunk-Based Development (Modern/DevOps)
Designed for Continuous Deployment (CI/CD).
-   **Branches**: `main` (Trunk), short-lived `feature/*`.
-   **Pros**: Fast, continuous integration.
-   **Cons**: Requires strong automated testing.

---

## 🔨 Hands-On Implementation

### Part 1: Simulating GitFlow 🌊

1.  **Setup:**
    Initialize a repo. Create `main` and `develop`.
    ```bash
    git init gitflow-lab
    cd gitflow-lab
    git checkout -b main
    git commit --allow-empty -m "Initial Commit"
    git checkout -b develop
    ```

2.  **Start Feature:**
    ```bash
    git checkout -b feature/login develop
    touch login.js
    git commit -am "Add login"
    ```

3.  **Finish Feature:**
    Merge into `develop` (NOT main).
    ```bash
    git checkout develop
    git merge feature/login
    git branch -d feature/login
    ```

4.  **Start Release:**
    We are ready for v1.0.
    ```bash
    git checkout -b release/1.0 develop
    # Bump version numbers, etc.
    git commit -am "Bump version to 1.0"
    ```

5.  **Finish Release:**
    Merge into `main` AND `develop`.
    ```bash
    git checkout main
    git merge release/1.0
    git tag v1.0
    
    git checkout develop
    git merge release/1.0
    ```

### Part 2: Simulating Trunk-Based 🌳

1.  **Setup:**
    Initialize a repo. Only `main` exists.
    ```bash
    cd ..
    git init trunk-lab
    cd trunk-lab
    git checkout -b main
    git commit --allow-empty -m "Initial Commit"
    ```

2.  **Developer A works:**
    Create short-lived branch.
    ```bash
    git checkout -b feat/user-api
    touch api.js
    git commit -am "Add API"
    ```

3.  **Merge immediately:**
    ```bash
    git checkout main
    git merge feat/user-api
    git branch -d feat/user-api
    ```

4.  **Feature Flags (The Secret Sauce):**
    How do you merge unfinished code? Hide it behind a flag.
    *Code Example:*
    ```javascript
    if (FEATURE_FLAGS.NEW_UI) {
        showNewUI();
    } else {
        showOldUI();
    }
    ```

---

## 🎯 Challenges

### Challenge 1: The Hotfix (GitFlow) (Difficulty: ⭐⭐⭐)

**Scenario:** v1.0 is live (on `main`). A critical bug is found.
**Task:**
1.  Create `hotfix/1.0.1` from `main`.
2.  Fix the bug.
3.  Merge into `main` AND `develop`.
4.  Tag `v1.0.1`.

### Challenge 2: Cherry Pick (Difficulty: ⭐⭐⭐)

**Scenario:** You fixed a bug in `develop` but you need that specific fix in `main` RIGHT NOW, without merging all the other unstable stuff in `develop`.
**Task:**
1.  Make a commit in `develop`.
2.  Switch to `main`.
3.  Use `git cherry-pick <commit-hash>` to copy just that one commit.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1 (Hotfix):**
```bash
git checkout main
git checkout -b hotfix/1.0.1
# Fix bug
git commit -am "Fix critical bug"
# Merge to main
git checkout main
git merge hotfix/1.0.1
git tag v1.0.1
# Merge to develop
git checkout develop
git merge hotfix/1.0.1
```

**Challenge 2 (Cherry Pick):**
```bash
git log develop # Get Hash
git checkout main
git cherry-pick <HASH>
```
</details>

---

## 🔑 Key Takeaways

1.  **DevOps prefers Trunk-Based**: It enables CI/CD. GitFlow delays integration.
2.  **Feature Flags**: Essential for Trunk-Based Development. Decouple "Deployment" from "Release".
3.  **Complexity Cost**: GitFlow adds overhead. Use it only if you maintain multiple versions of software (e.g., supporting v1.0 while building v2.0).

---

## ⏭️ Next Steps

We have the workflow. Now let's look at some advanced tools.

Proceed to **Lab 3.7: .gitignore and Stashing**.
