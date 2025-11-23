# Lab 3.4: Git Workflows

## Objective
Implement Git workflows for team collaboration.

## Learning Objectives
- Use GitFlow workflow
- Implement trunk-based development
- Manage feature branches
- Handle merge conflicts

---

## GitFlow Workflow

```bash
# Initialize GitFlow
git flow init

# Start feature
git flow feature start user-authentication

# Work on feature
git add .
git commit -m "Add user authentication"

# Finish feature
git flow feature finish user-authentication

# Start release
git flow release start 1.0.0

# Finish release
git flow release finish 1.0.0

# Hotfix
git flow hotfix start security-patch
git flow hotfix finish security-patch
```

## Trunk-Based Development

```bash
# Create short-lived branch
git checkout -b feature/quick-fix

# Make changes
git add .
git commit -m "Quick fix"

# Rebase on main
git fetch origin
git rebase origin/main

# Push and create PR
git push origin feature/quick-fix

# After review, squash merge
git checkout main
git merge --squash feature/quick-fix
git commit -m "Add quick fix"
```

## Conflict Resolution

```bash
# Merge conflict occurs
git merge feature-branch

# View conflicts
git status

# Resolve in file
# <<<<<<< HEAD
# Current changes
# =======
# Incoming changes
# >>>>>>> feature-branch

# Mark as resolved
git add conflicted-file.txt
git commit -m "Resolve merge conflict"
```

## Success Criteria
✅ GitFlow implemented  
✅ Trunk-based development used  
✅ Conflicts resolved  
✅ Team workflow established  

**Time:** 40 min
