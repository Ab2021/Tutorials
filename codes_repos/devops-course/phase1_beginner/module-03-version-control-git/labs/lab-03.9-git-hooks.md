# Lab 3.9: Git Hooks & Automation

## 🎯 Objective

Automate quality control using Git Hooks. You will create scripts that run automatically before you commit (`pre-commit`) or push (`pre-push`) to prevent bad code from entering the repository.

## 📋 Prerequisites

-   Completed Lab 3.8.
-   Basic Bash scripting knowledge.

## 📚 Background

### What are Hooks?
Scripts living in `.git/hooks/`.
-   **Client-Side**: Run on your machine (e.g., `pre-commit`, `commit-msg`).
-   **Server-Side**: Run on the server (e.g., `pre-receive`). *GitHub doesn't let you edit these directly, but Enterprise Git does.*

**Use Cases:**
-   Linting code.
-   Running unit tests.
-   Checking commit message format ("JIRA-123: Message").
-   Scanning for secrets (AWS Keys).

---

## 🔨 Hands-On Implementation

### Part 1: The Pre-Commit Hook (Linting) 🧹

**Goal:** Prevent committing if the code contains `TODO`.

1.  **Setup:**
    ```bash
    mkdir hooks-lab
    cd hooks-lab
    git init
    ```

2.  **Create Hook:**
    ```bash
    cd .git/hooks
    nano pre-commit
    ```
    Content:
    ```bash
    #!/bin/bash
    echo "Running pre-commit check..."
    
    # Grep for TODO in staged files
    if git diff --cached | grep -q "TODO"; then
        echo "❌ Error: You have a TODO in your code."
        exit 1
    fi
    
    echo "✅ Check passed."
    exit 0
    ```

3.  **Make Executable:**
    ```bash
    chmod +x pre-commit
    cd ../..
    ```

4.  **Test Failure:**
    ```bash
    echo "TODO: Fix this" > file.txt
    git add file.txt
    git commit -m "Test"
    ```
    *Result:* Commit blocked.

5.  **Test Success:**
    ```bash
    echo "Fixed code" > file.txt
    git add file.txt
    git commit -m "Test"
    ```
    *Result:* Commit succeeds.

### Part 2: The Commit-Msg Hook (Formatting) 📝

**Goal:** Enforce that commit messages start with "FEAT:", "FIX:", or "DOCS:".

1.  **Create Hook:**
    `.git/hooks/commit-msg`
    ```bash
    #!/bin/bash
    MSG_FILE=$1
    MSG_CONTENT=$(cat "$MSG_FILE")
    
    PATTERN="^(FEAT|FIX|DOCS):"
    
    if [[ ! "$MSG_CONTENT" =~ $PATTERN ]]; then
        echo "❌ Error: Commit message must start with FEAT:, FIX:, or DOCS:"
        exit 1
    fi
    ```

2.  **Make Executable:**
    `chmod +x .git/hooks/commit-msg`

3.  **Test:**
    `git commit --allow-empty -m "Bad message"` (Fails)
    `git commit --allow-empty -m "FEAT: Good message"` (Passes)

---

## 🎯 Challenges

### Challenge 1: Sharing Hooks (Difficulty: ⭐⭐⭐)

**Problem:** Hooks in `.git/hooks` are **not** pushed to the remote repo. Your teammates won't get them.
**Task:**
1.  Create a folder `.githooks` in your repo root.
2.  Move your hooks there.
3.  Configure Git to look there:
    `git config core.hooksPath .githooks`
4.  Commit the `.githooks` folder so teammates get it.

### Challenge 2: Husky (Difficulty: ⭐⭐⭐⭐)

**Task:**
If you have Node.js installed, research **Husky**. It's the industry standard for managing hooks in JS projects. Try to set it up.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
mkdir .githooks
mv .git/hooks/pre-commit .githooks/
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
git add .githooks
git commit -m "Add shared hooks"
```
Now, when teammates clone, they just run `git config core.hooksPath .githooks` once.
</details>

---

## 🔑 Key Takeaways

1.  **Automation**: Don't rely on memory ("Did I run the linter?"). Enforce it.
2.  **Bypassing**: You can bypass hooks with `git commit --no-verify` (use wisely!).
3.  **Shift Left**: Catching bugs at `pre-commit` is cheaper than catching them in CI, which is cheaper than Production.

---

## ⏭️ Next Steps

We have mastered Git. Let's put it all together.

Proceed to **Lab 3.10: Version Control Capstone**.
