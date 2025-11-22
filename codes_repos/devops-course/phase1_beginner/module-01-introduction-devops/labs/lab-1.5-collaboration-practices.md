# Lab 1.5: Collaboration Practices (Git & Code Review)

## 🎯 Objective

Move beyond basic "git add/commit" and master the collaborative workflows used in professional DevOps teams. You will simulate a multi-developer workflow, handle merge conflicts, create Pull Requests (PRs), and practice code reviews.

## 📋 Prerequisites

-   Completed Lab 1.4 (Git installed).
-   GitHub account (Free).
-   Basic understanding of Git commands.

## 🧰 Required Tools

-   **Git CLI**
-   **GitHub** (Web Interface)
-   **VS Code** (for editing and conflict resolution)

## 📚 Background

### The Pull Request (PR) Model

In a team, you rarely push directly to the `main` branch. Instead, you use the **Feature Branch Workflow**:

1.  **Branch**: Create a new branch for your feature (`feature/login-page`).
2.  **Commit**: Make changes in that branch.
3.  **Push**: Upload branch to remote (GitHub).
4.  **PR**: Open a Pull Request proposing to merge your branch into `main`.
5.  **Review**: Teammates review code, suggest changes, and approve.
6.  **Merge**: Code is merged into `main`.

This ensures quality control, runs automated tests (CI), and shares knowledge.

---

## 🔨 Hands-On Implementation

### Part 1: Setup Remote Repository 🌐

1.  **Create GitHub Repo:**
    -   Go to GitHub.com and create a new repository named `devops-collab-lab`.
    -   Initialize with a README.

2.  **Clone to Local:**
    ```bash
    git clone https://github.com/<your-username>/devops-collab-lab.git
    cd devops-collab-lab
    ```

3.  **Create a Base File:**
    Create `team_roster.txt` with a list of names.
    ```text
    Alice - Team Lead
    Bob - Developer
    ```
    Commit and push:
    ```bash
    git add team_roster.txt
    git commit -m "Initial roster"
    git push origin main
    ```

### Part 2: The Feature Branch Workflow 🌿

**Scenario:** You are "Developer A" adding a new member.

1.  **Create Branch:**
    ```bash
    git checkout -b feature/add-charlie
    ```

2.  **Make Changes:**
    Add "Charlie - DevOps" to `team_roster.txt`.

3.  **Commit and Push:**
    ```bash
    git add team_roster.txt
    git commit -m "Add Charlie to roster"
    git push -u origin feature/add-charlie
    ```

4.  **Create Pull Request:**
    -   Go to GitHub. You should see a "Compare & pull request" button.
    -   Click it. Title: "Add Charlie to team".
    -   Description: "Adding our new DevOps engineer."
    -   Click **Create Pull Request**.

5.  **Merge (Simulated):**
    -   Since you are the owner, you can merge your own PR.
    -   Click **Merge pull request** > **Confirm merge**.
    -   Delete the branch on GitHub.

6.  **Sync Local:**
    Back in your terminal:
    ```bash
    git checkout main
    git pull origin main
    # Verify Charlie is there
    cat team_roster.txt
    ```

### Part 3: Handling Merge Conflicts ⚔️

**Scenario:** Two developers edit the same line at the same time.

1.  **Create Branch 1 (Developer B):**
    ```bash
    git checkout -b feature/update-bob
    ```
    Change "Bob - Developer" to "Bob - Senior Developer" in `team_roster.txt`.
    Commit and push.

2.  **Create Branch 2 (Developer C):**
    *Go back to main first!*
    ```bash
    git checkout main
    git checkout -b feature/bob-promotion
    ```
    Change "Bob - Developer" to "Bob - Tech Lead" in `team_roster.txt`.
    Commit and push.

3.  **Merge Branch 1:**
    -   Go to GitHub. Create PR for `feature/update-bob`.
    -   Merge it.

4.  **Attempt to Merge Branch 2:**
    -   Create PR for `feature/bob-promotion`.
    -   **GitHub will warn:** "Can't automatically merge."

5.  **Resolve Conflict Locally:**
    ```bash
    git checkout feature/bob-promotion
    git pull origin main
    # Git will say: CONFLICT (content): Merge conflict in team_roster.txt
    ```

6.  **Fix in VS Code:**
    -   Open `team_roster.txt`.
    -   You will see markers:
        ```text
        <<<<<<< HEAD
        Bob - Tech Lead
        =======
        Bob - Senior Developer
        >>>>>>> main
        ```
    -   Decide the correct title (e.g., "Bob - Senior Tech Lead").
    -   Remove markers.

7.  **Finalize Merge:**
    ```bash
    git add team_roster.txt
    git commit -m "Resolve conflict: Bob is Senior Tech Lead"
    git push origin feature/bob-promotion
    ```

8.  **Merge on GitHub:**
    -   Refresh the PR page. It should now say "Able to merge".
    -   Merge it.

### Part 4: Code Review Best Practices 🧐

**Objective:** Learn what to look for in a review.

**Checklist for Reviewers:**
1.  **Functionality:** Does the code do what it says?
2.  **Style:** Does it follow team conventions?
3.  **Tests:** Are there tests? Do they pass?
4.  **Security:** Any hardcoded secrets? SQL injection risks?
5.  **Clarity:** Is the code readable? Comments where needed?

**Exercise:**
Create a file `review_checklist.md` in your repo with the above points and commit it directly to main (simulating a hotfix).

---

## 🎯 Challenges

### Challenge 1: The Rebase Workflow (Difficulty: ⭐⭐⭐)

**Background:** `git merge` creates a "merge commit". `git rebase` rewrites history to make it linear. Some teams prefer rebase.

**Task:**
1.  Create a branch `feature/rebase-test`.
2.  Make a commit.
3.  Update `main` with a different commit.
4.  Instead of `git merge main` into your branch, use `git rebase main`.
5.  Push the branch. *Hint: You might need `--force`.*

### Challenge 2: Git Hooks (Difficulty: ⭐⭐⭐⭐)

**Background:** Git hooks are scripts that run automatically before/after git events.

**Task:**
1.  Navigate to `.git/hooks/` in your repo.
2.  Create a `pre-commit` file (make it executable).
3.  Write a script that prevents committing if the file contains the word "PASSWORD".
4.  Test it by trying to commit a file with "PASSWORD" in it.

---

## 💡 Solution

<details>
<summary>Click to reveal Challenge 2 Solution (Git Hooks)</summary>

**File:** `.git/hooks/pre-commit` (No extension)

```bash
#!/bin/bash
# Pre-commit hook to check for secrets

echo "🔒 Running pre-commit security check..."

# Grep for "PASSWORD" in staged files
# git diff --cached --name-only lists staged files
if git diff --cached --name-only | xargs grep -q "PASSWORD"; then
    echo "❌ ERROR: Security Check Failed!"
    echo "   Found 'PASSWORD' in staged files."
    echo "   Please remove secrets before committing."
    exit 1
fi

echo "✅ Security Check Passed."
exit 0
```

**Make executable (Linux/Mac):** `chmod +x .git/hooks/pre-commit`
**Windows:** Git Bash usually handles the shebang, but ensure file is ASCII.

</details>

---

## 🔑 Key Takeaways

1.  **Communication**: Pull Requests are about communication, not just code merging.
2.  **Conflicts are Normal**: Don't panic. Read the markers, talk to the other developer, resolve.
3.  **Sync Often**: The longer you wait to pull from `main`, the harder the conflict resolution.
4.  **Protection**: Use Branch Protection Rules in GitHub (Settings > Branches) to prevent pushing directly to main.

---

## ⏭️ Next Steps

Now that we can collaborate, let's look at why we automate.

Proceed to **Lab 1.6: Automation Benefits**.
