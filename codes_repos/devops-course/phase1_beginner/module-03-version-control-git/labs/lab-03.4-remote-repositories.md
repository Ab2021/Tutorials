# Lab 3.4: Remote Repositories (GitHub)

## 🎯 Objective

Connect your local repository to a remote server (GitHub). Learn to push, pull, clone, and manage remote URLs.

## 📋 Prerequisites

-   Completed Lab 3.3.
-   GitHub Account.

## 📚 Background

### Local vs Remote

-   **Local**: On your laptop. Fast. Private.
-   **Remote**: On GitHub/GitLab. Shared. Backed up.
-   **Origin**: The default nickname for the remote server.

---

## 🔨 Hands-On Implementation

### Part 1: Connecting to GitHub 🔗

1.  **Create Repo on GitHub:**
    -   Click "+" -> "New Repository".
    -   Name: `devops-lab-remote`.
    -   **Do not** initialize with README (we want an empty repo).

2.  **Add Remote:**
    In your local terminal (inside a git repo):
    ```bash
    git remote add origin https://github.com/<YOUR_USERNAME>/devops-lab-remote.git
    ```

3.  **Verify:**
    ```bash
    git remote -v
    ```

### Part 2: Pushing 🚀

1.  **Push Main:**
    ```bash
    git branch -M main  # Ensure branch is named main
    git push -u origin main
    ```
    -   `-u`: Sets "Upstream". Next time you can just type `git push`.

2.  **Check GitHub:**
    Refresh the page. Your code is there!

### Part 3: Cloning 🐑

**Scenario:** You got a new laptop. You need your code.

1.  **Go to a different folder:**
    ```bash
    cd ..
    mkdir new-laptop
    cd new-laptop
    ```

2.  **Clone:**
    ```bash
    git clone https://github.com/<YOUR_USERNAME>/devops-lab-remote.git
    ```
    *Result:* It downloads the entire history.

### Part 4: Pulling ⬇️

**Scenario:** You made changes on the "new laptop" and pushed them. Now you are back on the "old laptop".

1.  **Simulate Change:**
    (In `new-laptop` folder)
    ```bash
    cd devops-lab-remote
    touch laptop2.txt
    git add .
    git commit -m "Work from laptop 2"
    git push
    ```

2.  **Update Old Laptop:**
    (Go back to original folder)
    ```bash
    git pull
    ```
    *Result:* `laptop2.txt` appears.

---

## 🎯 Challenges

### Challenge 1: The "Reject" (Difficulty: ⭐⭐)

**Scenario:**
1.  Change a file on GitHub (using the web editor).
2.  Change the *same* file locally.
3.  Try to `git push`.
4.  **Task:** Fix the error. (Hint: You must pull before you push).

### Challenge 2: SSH Keys (Difficulty: ⭐⭐⭐)

**Task:**
Switch your remote URL from HTTPS to SSH.
1.  Generate SSH key (Lab 2.9).
2.  Add public key to GitHub Settings.
3.  `git remote set-url origin git@github.com:...`
4.  Push without typing a password.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Git will say "Updates were rejected".
Fix:
```bash
git pull
# Resolve any conflicts
git push
```

**Challenge 2:**
```bash
git remote set-url origin git@github.com:<USER>/repo.git
```
</details>

---

## 🔑 Key Takeaways

1.  **Pull before Push**: Always get the latest changes before sending yours.
2.  **Origin is just a name**: You can have multiple remotes (e.g., `heroku`, `gitlab`).
3.  **Clone vs Init**: Use `clone` for existing projects, `init` for new ones.

---

## ⏭️ Next Steps

We can sync code. Now let's learn the advanced "Undo" buttons.

Proceed to **Lab 3.5: Undoing Changes (Reset, Revert)**.
