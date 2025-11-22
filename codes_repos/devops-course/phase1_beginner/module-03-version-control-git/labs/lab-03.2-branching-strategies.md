# Lab 3.2: Branching Strategies

## 🎯 Objective

Understand why and how to use branches. Learn to create, switch, merge, and delete branches. Explore the "Git Flow" concept.

## 📋 Prerequisites

-   Completed Lab 3.1.

## 📚 Background

### Why Branch?

Imagine you are building a website. It's live (Production). You want to add a "Dark Mode" feature, but it will take a week. You can't break the live site while working.
**Solution:** Create a copy (Branch), work there, and merge it back when done.

**Main Branch**: The "Truth". Always stable.
**Feature Branch**: Where work happens.

---

## 🔨 Hands-On Implementation

### Part 1: Creating & Switching 🌿

1.  **Check current branch:**
    ```bash
    git branch
    # Output: * main
    ```

2.  **Create a new branch:**
    ```bash
    git branch feature/login
    ```

3.  **Switch to it:**
    ```bash
    git checkout feature/login
    # OR (Newer Git)
    git switch feature/login
    ```

4.  **Shortcut (Create & Switch):**
    ```bash
    git checkout -b feature/logout
    ```

### Part 2: Diverging History 🔀

1.  **Make a commit on the feature branch:**
    (Ensure you are on `feature/logout`)
    ```bash
    touch logout.js
    git add logout.js
    git commit -m "Add logout feature"
    ```

2.  **Switch back to main:**
    ```bash
    git checkout main
    ```
    *Observe:* `logout.js` disappears! It only exists in the other branch.

3.  **Make a commit on main (Simulate teammate's work):**
    ```bash
    touch readme.md
    git add readme.md
    git commit -m "Add documentation"
    ```

### Part 3: Merging 🤝

We want to bring the "Logout" feature into Main.

1.  **Go to target branch (Main):**
    ```bash
    git checkout main
    ```

2.  **Merge source branch:**
    ```bash
    git merge feature/logout
    ```
    *Output:* A "Merge Commit" is created joining the two histories.

3.  **Delete the feature branch:**
    (Clean up after yourself)
    ```bash
    git branch -d feature/logout
    ```

---

## 🎯 Challenges

### Challenge 1: The Fast-Forward Merge (Difficulty: ⭐⭐)

**Task:**
1.  Create a branch `quick-fix`.
2.  Commit a change.
3.  Merge it into `main`.
4.  Notice that Git says "Fast-forward".
5.  **Question:** Why didn't it create a merge commit?

### Challenge 2: Visualizing the Graph (Difficulty: ⭐⭐)

**Task:**
Use `git log` with options to see the branch structure visually in the terminal.
*Hint: `git log --graph --oneline --all`*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
"Fast-forward" happens when `main` hasn't changed since you created the branch. Git just moves the `main` pointer forward to your latest commit. No divergence = No merge commit needed.

**Challenge 2:**
```bash
git log --graph --oneline --all --decorate
```
</details>

---

## 🔑 Key Takeaways

1.  **Never commit to main**: Always use a feature branch.
2.  **Delete merged branches**: Keep your repo clean.
3.  **Checkout vs Switch**: `git switch` is the modern, less confusing version of `git checkout` for branches.

---

## ⏭️ Next Steps

Merging is easy when files don't overlap. But what if they do?

Proceed to **Lab 3.3: Merge Conflicts**.
