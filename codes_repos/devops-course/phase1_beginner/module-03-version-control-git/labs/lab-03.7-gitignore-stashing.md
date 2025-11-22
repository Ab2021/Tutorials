# Lab 3.7: .gitignore and Stashing

## 🎯 Objective

Learn how to keep your repository clean. You will configure `.gitignore` to exclude secrets and build artifacts, and use `git stash` to save work temporarily without committing.

## 📋 Prerequisites

-   Completed Lab 3.6.

## 📚 Background

### Ignoring Files
Not everything belongs in Git.
-   **Yes**: Source code, config templates, documentation.
-   **No**: Passwords (`.env`), Build artifacts (`/bin`, `/dist`), OS files (`.DS_Store`), Dependencies (`node_modules`).

### Stashing
Imagine you are working on a messy feature. Your boss asks you to fix a critical bug *right now*. You can't commit (code is broken). You can't switch branches (Git won't let you).
**Solution:** Stash it. (Put it in a temporary drawer).

---

## 🔨 Hands-On Implementation

### Part 1: The .gitignore File 🙈

1.  **Setup:**
    Create a repo with some "junk".
    ```bash
    mkdir ignore-lab
    cd ignore-lab
    git init
    touch main.py
    touch secret.key
    mkdir logs
    touch logs/app.log
    ```

2.  **Create .gitignore:**
    ```bash
    nano .gitignore
    ```
    Content:
    ```text
    # Secrets
    *.key
    
    # Logs
    logs/
    
    # OS Files
    .DS_Store
    ```

3.  **Verify:**
    ```bash
    git status
    ```
    *Output:* Git should ONLY see `main.py` and `.gitignore`. It should NOT see `secret.key` or `logs/`.

4.  **Commit:**
    ```bash
    git add .
    git commit -m "Initial commit with ignore rules"
    ```

### Part 2: Stashing Changes 📦

1.  **Start Work:**
    ```bash
    echo "Working on feature..." >> main.py
    ```

2.  **Interruption:**
    Try to switch branches (simulate).
    ```bash
    git checkout -b hotfix
    ```
    *Note:* Sometimes Git allows this if no conflicts, but let's assume we want a clean slate.

3.  **Stash:**
    ```bash
    git stash
    ```
    *Output:* `Saved working directory and index state...`
    *Check file:* `cat main.py` (Changes are gone!).

4.  **Do other work:**
    ```bash
    touch hotfix.txt
    git add .
    git commit -m "Fix bug"
    ```

5.  **Retrieve Stash:**
    ```bash
    git stash pop
    ```
    *Result:* Changes are back in `main.py`.

---

## 🎯 Challenges

### Challenge 1: Ignoring Tracked Files (Difficulty: ⭐⭐⭐)

**Scenario:** You accidentally committed `secret.key` *before* creating `.gitignore`. Now you added it to `.gitignore`, but Git is *still* tracking it.
**Task:**
Tell Git to "forget" the file from the index, but keep it on your disk.
*Hint: `git rm --cached`*

### Challenge 2: Multiple Stashes (Difficulty: ⭐⭐)

**Task:**
1.  Make change A. Stash it.
2.  Make change B. Stash it.
3.  List stashes (`git stash list`).
4.  Apply specific stash (Change A).
    *Hint: `git stash apply stash@{1}`*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
git rm --cached secret.key
git commit -m "Stop tracking secret key"
```
*Note:* The key is still in history! You need BFG Repo-Cleaner to remove it permanently.

**Challenge 2:**
```bash
git stash list
git stash apply stash@{1}
```
</details>

---

## 🔑 Key Takeaways

1.  **Global Ignore**: You can set a global `.gitignore` for all your projects (e.g., for `.DS_Store`).
2.  **Pop vs Apply**: `pop` applies and deletes the stash. `apply` applies but keeps it in the list.
3.  **Security**: `.gitignore` does NOT delete files from history. If you commit a password, consider it compromised. Change the password.

---

## ⏭️ Next Steps

Let's get advanced.

Proceed to **Lab 3.8: Advanced Git (Rebase, Squash, Bisect)**.
