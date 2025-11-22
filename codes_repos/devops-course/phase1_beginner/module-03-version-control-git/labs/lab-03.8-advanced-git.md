# Lab 3.8: Advanced Git (Rebase, Squash, Bisect)

## 🎯 Objective

Level up your Git skills. Learn to rewrite history for a clean log (`rebase`, `squash`) and find bugs automatically (`bisect`).

## 📋 Prerequisites

-   Completed Lab 3.7.

## 📚 Background

### Merge vs Rebase
-   **Merge**: Preserves history exactly as it happened. (Messy, truthful).
-   **Rebase**: Rewrites history to look linear. (Clean, "lie").

### Squashing
Combining 10 small commits ("typo", "fix", "wip") into 1 clean commit ("Add Feature X").

### Bisect
Binary search for bugs. "It worked in v1.0, it's broken in v2.0. Which of the 100 commits in between broke it?"

---

## 🔨 Hands-On Implementation

### Part 1: Interactive Rebase (Squashing) 🥞

1.  **Setup:**
    ```bash
    mkdir advanced-git
    cd advanced-git
    git init
    touch file.txt
    git add .
    git commit -m "Init"
    ```

2.  **Make messy commits:**
    ```bash
    echo "A" >> file.txt; git commit -am "WIP"
    echo "B" >> file.txt; git commit -am "Fix typo"
    echo "C" >> file.txt; git commit -am "Done"
    ```

3.  **Squash them:**
    We want to combine the last 3 commits.
    ```bash
    git rebase -i HEAD~3
    ```
    *Interactive Editor opens:*
    Change `pick` to `squash` (or `s`) for the 2nd and 3rd lines.
    Leave the 1st line as `pick`.
    
    ```text
    pick a1b2c WIP
    squash d3e4f Fix typo
    squash g5h6i Done
    ```
    Save and close.
    Git asks for a new commit message. Type: "Add Feature ABC".

4.  **Verify:**
    `git log` shows only 1 commit instead of 3.

### Part 2: Git Bisect 🕵️‍♂️

**Scenario:** We have 10 commits. Commit 5 introduced a bug. We don't know which one.

1.  **Create History:**
    Create a script `generate_commits.sh`:
    ```bash
    for i in {1..10}; do
        echo "Line $i" >> code.txt
        if [ $i -eq 5 ]; then echo "BUG" >> code.txt; fi
        git commit -am "Commit $i"
    done
    ```
    Run it.

2.  **Start Bisect:**
    ```bash
    git bisect start
    git bisect bad HEAD   # Current version is bad
    git bisect good HEAD~10 # 10 commits ago was good
    ```

3.  **The Search:**
    Git jumps to the middle (Commit 5 or 6).
    Check the file: `cat code.txt`.
    -   If "BUG" is there: `git bisect bad`
    -   If not: `git bisect good`
    
    Repeat until Git says: `Commit X is the first bad commit`.

4.  **Reset:**
    ```bash
    git bisect reset
    ```

---

## 🎯 Challenges

### Challenge 1: Rebase Conflicts (Difficulty: ⭐⭐⭐⭐)

**Task:**
1.  Create branch `feat`. Make a change.
2.  Update `main` with a conflicting change.
3.  Rebase `feat` onto `main` (`git checkout feat; git rebase main`).
4.  Resolve the conflict during the rebase.
    *Hint: Fix file, `git add`, `git rebase --continue`.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
When conflict happens:
1.  Open file, fix conflict markers.
2.  `git add filename`
3.  `git rebase --continue`
4.  (Do NOT run `git commit`).
</details>

---

## 🔑 Key Takeaways

1.  **Never Rebase Public Branches**: If you rebase `main` and push, you break everyone else's repo. Only rebase your local feature branches.
2.  **Squash before Merge**: Makes the main history clean and readable.
3.  **Bisect is Magic**: Saves hours of manual checking.

---

## ⏭️ Next Steps

We are Git masters. Now let's automate the CI/CD pipeline.

Proceed to **Lab 3.9: Git Hooks & Automation**.
