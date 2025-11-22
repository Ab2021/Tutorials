# Version Control with Git

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of version control with Git, including:
- **Architecture**: Understanding the Object Database, References, and the Three Trees.
- **Core Commands**: Mastering the daily workflow (`add`, `commit`, `push`, `pull`).
- **Branching**: Implementing effective strategies (Git Flow, Trunk-Based).
- **Collaboration**: resolving conflicts and reviewing code via Pull Requests.
- **Advanced Techniques**: Using `rebase`, `bisect`, and `cherry-pick` to maintain a clean history.

---

## 📖 Theoretical Concepts

### 1. Git Fundamentals and History

Git is a **Distributed Version Control System (DVCS)** created by Linus Torvalds in 2005 to manage the Linux kernel development. Unlike centralized systems (SVN, CVS) where history lives on a central server, every Git repository is a full backup of the entire history.

**Key Concepts:**
- **Distributed**: Every developer has a full copy of the repo. You can work offline.
- **Snapshots, Not Differences**: Git thinks of data like a stream of snapshots. If a file hasn't changed, Git just links to the previous identical file.
- **Checksums**: Everything is checksummed (SHA-1) before it is stored. It is impossible to change the contents of any file or directory without Git knowing about it.

### 2. Git Architecture (The Three Trees)

Understanding Git requires understanding the three states your files can reside in:

1.  **Working Directory**: The actual files on your disk that you edit.
2.  **Staging Area (Index)**: A file that stores information about what will go into your next commit.
3.  **Repository (HEAD)**: The committed history (snapshots) stored in the `.git` directory.

**The Workflow:**
1.  **Modify** files in your working directory.
2.  **Stage** the files, adding snapshots of them to your staging area (`git add`).
3.  **Commit**, which takes the files as they are in the staging area and stores that snapshot permanently to your Git directory (`git commit`).

### 3. Branching and Merging Strategies

Branching means you diverge from the main line of development and continue to do work without messing with that main line. In Git, a branch is simply a lightweight movable pointer to a commit.

**Strategies:**
- **Git Flow**: Strict branching model with `develop`, `master`, `feature/`, `release/`, and `hotfix/` branches. Good for traditional release cycles.
- **GitHub Flow**: Simple workflow. `main` is always deployable. Create feature branches, open PRs, merge to `main`. Good for CI/CD.
- **Trunk-Based Development**: Developers merge small, frequent updates to a core "trunk" (main). Requires Feature Flags to hide unfinished work.

**Merge vs. Rebase:**
- **Merge**: Creates a new "merge commit". Preserves history exactly as it happened.
- **Rebase**: Moves the entire feature branch to begin on the tip of the main branch. Creates a linear history but rewrites commit hashes. **Golden Rule: Never rebase public branches.**

### 4. Advanced Git

- **Cherry-Pick**: Apply the changes introduced by some existing commits. Useful for backporting hotfixes.
- **Bisect**: Use binary search to find the commit that introduced a bug.
- **Reflog**: A safety net. It records when the tip of branches were updated. You can recover "lost" commits using reflog.
- **Stash**: Temporarily shelve (or stash) changes you've made to your working copy so you can work on something else, and then come back and re-apply them later.

---

## 🔧 Practical Examples

### Basic Workflow

```bash
# Initialize
git init

# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "feat: add login page"

# View history
git log --oneline --graph --all
```

### Advanced: Interactive Rebase

Clean up your local history before merging.

```bash
# Rebase last 3 commits
git rebase -i HEAD~3
```

*In the editor, change `pick` to `squash` to combine commits.*

### Finding a Bug with Bisect

```bash
git bisect start
git bisect bad            # Current version is bad
git bisect good v1.0      # Version 1.0 was good
# Git checks out a middle commit
# Test your app...
git bisect good           # If this version works
# Git checks out the next half...
```

---

## 🎯 Hands-on Labs

- [Lab 3.1: Git Basics (Init, Add, Commit)](./labs/lab-03.1-git-basics.md)
- [Lab 3.10: Version Control Capstone Project](./labs/lab-03.10-git-project.md)
- [Lab 3.2: Branching Strategies](./labs/lab-03.2-branching-strategies.md)
- [Lab 3.3: Handling Merge Conflicts](./labs/lab-03.3-merge-conflicts.md)
- [Lab 3.4: Remote Repositories (GitHub)](./labs/lab-03.4-remote-repositories.md)
- [Lab 3.5: Undoing Changes (Reset, Revert, Checkout)](./labs/lab-03.5-undoing-changes.md)
- [Lab 3.6: Git Workflows (GitFlow vs Trunk-Based)](./labs/lab-03.6-git-workflows.md)
- [Lab 3.7: .gitignore and Stashing](./labs/lab-03.7-gitignore-stashing.md)
- [Lab 3.8: Advanced Git (Rebase, Squash, Bisect)](./labs/lab-03.8-advanced-git.md)
- [Lab 3.9: Git Hooks & Automation](./labs/lab-03.9-git-hooks.md)

---

## 📚 Additional Resources

### Official Documentation
- [Pro Git Book (Free)](https://git-scm.com/book/en/v2)
- [GitHub Skills](https://skills.github.com/)

### Interactive Tutorials
- [Learn Git Branching](https://learngitbranching.js.org/) - Highly recommended visualizer.
- [Oh My Git!](https://ohmygit.org/) - An open source game about learning Git.

---

## 🔑 Key Takeaways

1.  **Commit Often**: Small, atomic commits are easier to review and revert.
2.  **Write Good Messages**: Subject line (50 chars), Body (Why, not What).
3.  **Don't Panic**: Almost nothing is lost in Git. `git reflog` is your friend.
4.  **Pull Requests**: Use them for code review, not just merging.

---

## ⏭️ Next Steps

1.  Complete the labs to practice these concepts.
2.  Proceed to **[Module 4: Networking Basics](../module-04-networking-basics/README.md)** to understand the network layer underlying DevOps.
