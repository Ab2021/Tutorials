# Lab 3.5: Undoing Changes (Reset, Revert, Checkout)

## 🎯 Objective

Learn how to fix mistakes in Git. You will learn the difference between "soft" undos (keeping work), "hard" undos (destroying work), and "safe" undos (reverting public history).

## 📋 Prerequisites

-   Completed Lab 3.4.

## 📚 Background

### The Time Machine

Git allows you to travel back in time.
-   **Checkout**: Look at the past (Read-only).
-   **Revert**: Fix the past by adding a new commit (Safe for public).
-   **Reset**: Rewrite the past (Dangerous for public).

---

## 🔨 Hands-On Implementation

### Part 1: Discarding Local Changes (`checkout / restore`) 🗑️

**Scenario:** You edited a file, broke it, and want to go back to the last commit.

1.  **Mess up a file:**
    ```bash
    echo "Bad code" >> hello.txt
    ```

2.  **Discard changes:**
    ```bash
    git checkout hello.txt
    # OR
    git restore hello.txt
    ```
    *Result:* `hello.txt` is back to normal.

### Part 2: Unstaging (`reset`) 🔙

**Scenario:** You ran `git add` but didn't mean to.

1.  **Stage a file:**
    ```bash
    echo "More changes" >> hello.txt
    git add hello.txt
    ```

2.  **Unstage:**
    ```bash
    git reset HEAD hello.txt
    # OR
    git restore --staged hello.txt
    ```
    *Result:* File is modified but not staged.

### Part 3: The "Soft" Reset (Undo Commit, Keep Work) ↩️

**Scenario:** You committed "WIP" but want to add more to it.

1.  **Commit:**
    ```bash
    git commit -am "WIP: Started feature"
    ```

2.  **Soft Reset:**
    ```bash
    git reset --soft HEAD~1
    ```
    *Result:* The commit is gone. The changes are back in your Staging Area. You can add more and commit again.

### Part 4: The "Hard" Reset (Destroy Work) 💥

**Scenario:** You want to burn everything since the last commit.

1.  **Make changes:**
    ```bash
    touch junk.txt
    echo "Trash" > hello.txt
    git add .
    git commit -m "Garbage commit"
    ```

2.  **Hard Reset:**
    ```bash
    git reset --hard HEAD~1
    ```
    *Result:* `junk.txt` is deleted. `hello.txt` is reverted. **Data is lost forever.**

### Part 5: Revert (The Safe Undo) 🛡️

**Scenario:** You pushed a bug to production. You can't use `reset` because others have pulled the code.

1.  **Create a bug:**
    ```bash
    echo "Bug" >> production.code
    git add .
    git commit -m "Deploy feature X"
    git push
    ```

2.  **Revert:**
    ```bash
    git revert HEAD
    ```
    *Result:* Git creates a *new* commit that does the exact opposite of the bad commit.
    *Message:* `Revert "Deploy feature X"`

3.  **Push:**
    ```bash
    git push
    ```
    *History:* `Deploy` -> `Revert`. History is preserved.

---

## 🎯 Challenges

### Challenge 1: Recovering a Deleted File (Difficulty: ⭐⭐)

**Task:**
1.  Delete a tracked file: `rm hello.txt`.
2.  Recover it using Git.

### Challenge 2: The Reflog (Difficulty: ⭐⭐⭐⭐)

**Scenario:** You did a `git reset --hard` and realized you made a mistake. You want that "Garbage commit" back.
**Task:**
1.  Run `git reflog`.
2.  Find the SHA of the commit *before* the reset.
3.  Reset to that SHA.
    *Hint: Git keeps a secret log of everywhere HEAD has been.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
git checkout HEAD hello.txt
# OR
git restore hello.txt
```

**Challenge 2:**
```bash
git reflog
# Find hash, e.g., a1b2c3d HEAD@{1}: commit: Garbage commit
git reset --hard a1b2c3d
```
</details>

---

## 🔑 Key Takeaways

1.  **Public vs Private**: Use `reset` only on local branches. Use `revert` on shared branches (main).
2.  **Reflog saves lives**: If you think you lost everything, check `git reflog`.
3.  **Checkout is overloaded**: `git checkout` does too many things (branches, files). Use `git switch` (branches) and `git restore` (files) if you have a new Git version.

---

## ⏭️ Next Steps

We know the commands. Now let's learn the workflows used by teams.

Proceed to **Lab 3.6: Git Workflows (GitFlow vs Trunk)**.
