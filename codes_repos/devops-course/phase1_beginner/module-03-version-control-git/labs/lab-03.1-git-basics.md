# Lab 3.1: Git Basics (Init, Add, Commit)

## 🎯 Objective

Master the fundamental Git workflow. You will initialize a repository, track files, make commits, and understand the three states of Git (Working Directory, Staging Area, Repository).

## 📋 Prerequisites

-   Git installed (from Lab 1.4).
-   VS Code installed.

## 📚 Background

### The Three States

1.  **Working Directory**: Where you edit files. (Untracked/Modified)
2.  **Staging Area (Index)**: Where you prepare files for a commit. (Staged)
3.  **Repository (.git)**: Where Git permanently stores the snapshot. (Committed)

**The Workflow:**
`Edit` -> `git add` -> `Staging` -> `git commit` -> `Repo`

---

## 🔨 Hands-On Implementation

### Part 1: Initialization 🏁

1.  **Create a project folder:**
    ```bash
    mkdir git-lab-3
    cd git-lab-3
    ```

2.  **Initialize Git:**
    ```bash
    git init
    ```
    *Output:* `Initialized empty Git repository in .../.git/`
    *Note:* The `.git` folder contains all the history. Don't delete it!

### Part 2: The First Commit 🥇

1.  **Create a file:**
    ```bash
    echo "Hello Git" > hello.txt
    ```

2.  **Check Status:**
    ```bash
    git status
    ```
    *Output:* `Untracked files: hello.txt` (Red color usually).

3.  **Stage the file:**
    ```bash
    git add hello.txt
    ```

4.  **Check Status again:**
    ```bash
    git status
    ```
    *Output:* `Changes to be committed: hello.txt` (Green color).

5.  **Commit:**
    ```bash
    git commit -m "Initial commit"
    ```
    *Output:* `[main (root-commit) ...] Initial commit`

### Part 3: The History (`git log`) 📜

1.  **View Log:**
    ```bash
    git log
    ```
    *Shows:* Commit Hash, Author, Date, Message.

2.  **One-line Log (Cleaner):**
    ```bash
    git log --oneline
    ```

### Part 4: Making Changes ✏️

1.  **Modify the file:**
    ```bash
    echo "Adding a second line" >> hello.txt
    ```

2.  **Check Diff:**
    See what changed *before* you stage.
    ```bash
    git diff
    ```

3.  **Stage and Commit:**
    ```bash
    git add hello.txt
    git commit -m "Update hello.txt"
    ```

---

## 🎯 Challenges

### Challenge 1: The "Oops" (Difficulty: ⭐⭐)

**Scenario:** You staged a file but didn't mean to.
**Task:**
1.  Create `mistake.txt`.
2.  Stage it (`git add mistake.txt`).
3.  **Unstage** it without deleting the file.
    *Hint: Read the output of `git status` carefully.*

### Challenge 2: Amending (Difficulty: ⭐⭐⭐)

**Scenario:** You made a commit but forgot to add a file, or made a typo in the message.
**Task:**
1.  Make a commit.
2.  Change the commit message *without* making a new commit.
    *Hint: `git commit --amend`*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
git reset HEAD mistake.txt
# OR (Newer Git versions)
git restore --staged mistake.txt
```

**Challenge 2:**
```bash
git commit --amend -m "New correct message"
```
</details>

---

## 🔑 Key Takeaways

1.  **Commit Often**: Small, frequent commits are better than one giant commit.
2.  **Meaningful Messages**: "Fix bug" is bad. "Fix login timeout issue" is good.
3.  **Status is your friend**: Run `git status` constantly to know where you are.

---

## ⏭️ Next Steps

We can save history linearly. Now let's learn to branch out.

Proceed to **Lab 3.2: Branching Strategies**.
