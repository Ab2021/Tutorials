# Git Initialization and GitHub Push Workflow

This guide documents how to initialize the handbook folder as a Git repository, create and check out the `FInetuning` branch, and push it to GitHub.

## 1. Open the project folder

```powershell
cd D:\Finetuning\Finetuning-Handbook
```

## 2. Initialize Git

```powershell
git init
```

## 3. Create and check out the branch

```powershell
git checkout -b FInetuning
```

To check the active branch:

```powershell
git branch --show-current
```

## 4. Add a `.gitignore`

For Python projects, ignore generated cache files:

```gitignore
__pycache__/
*.py[cod]
```

## 5. Stage and commit the files

```powershell
git add .
git commit -m "Initial Finetuning handbook"
```

If files are changed later, commit them with:

```powershell
git add .
git commit -m "Update finetuning handbook"
```

## 6. Add the GitHub remote

```powershell
git remote add origin https://github.com/Ab2021/Tutorials.git
```

Verify the remote:

```powershell
git remote -v
```

## 7. Push the branch

```powershell
git push -u origin FInetuning
```

The `-u` option links the local branch to the remote branch, so future pushes can use:

```powershell
git push
```

## 8. Verify the final state

```powershell
git status --short --branch
git branch --show-current
```

The active branch should be `FInetuning`, and the branch should track `origin/FInetuning`.
