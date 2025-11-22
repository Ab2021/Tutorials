# Lab 06.2: GitHub Actions Basics - Your First Pipeline

## Objective
Create your first CI/CD pipeline using GitHub Actions to automatically test a simple application on every commit.

## Prerequisites
- GitHub account (free tier)
- Git installed locally
- Basic knowledge of Git commands
- Completed Lab 06.1 (CI/CD Concepts)

## Learning Objectives
- Create a GitHub Actions workflow file
- Understand YAML syntax for workflows
- Trigger automated builds on commits
- View workflow execution logs
- Debug failed workflows

---

## Part 1: Understanding GitHub Actions

### What is GitHub Actions?

GitHub Actions is a CI/CD platform integrated directly into GitHub. It allows you to:
- Run automated workflows when events occur (push, pull request, schedule)
- Execute jobs in parallel or sequentially
- Use pre-built actions from the marketplace
- Run on GitHub-hosted or self-hosted runners

### Key Concepts

| Term | Definition | Example |
|------|------------|---------|
| **Workflow** | Automated process defined in YAML | `.github/workflows/ci.yml` |
| **Event** | Trigger that starts a workflow | `push`, `pull_request` |
| **Job** | Set of steps that run on the same runner | `build`, `test`, `deploy` |
| **Step** | Individual task within a job | Run a command, use an action |
| **Runner** | Server that executes workflows | `ubuntu-latest`, `windows-latest` |

---

## Part 2: Setup - Create a Sample Project

### Step 1: Create a New Repository

```bash
# Create a new directory
mkdir my-first-pipeline
cd my-first-pipeline

# Initialize Git
git init

# Create a simple Python application
cat > app.py << 'EOF'
def add(a, b):
    """Add two numbers"""
    return a + b

def subtract(a, b):
    """Subtract two numbers"""
    return a - b

if __name__ == "__main__":
    print(f"2 + 3 = {add(2, 3)}")
    print(f"5 - 2 = {subtract(5, 2)}")
EOF

# Create a test file
cat > test_app.py << 'EOF'
from app import add, subtract

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(5, 2) == 3
    assert subtract(10, 10) == 0
    assert subtract(0, 5) == -5

print("All tests passed!")
EOF

# Create requirements file
echo "pytest==7.4.0" > requirements.txt

# Test locally (optional)
python test_app.py
```

**Expected Output:**
```
All tests passed!
```

### Step 2: Create GitHub Repository

1. Go to [github.com](https://github.com) and click "New Repository"
2. Name it `my-first-pipeline`
3. Keep it **Public** (private repos have limited free Actions minutes)
4. **Do NOT** initialize with README
5. Click "Create repository"

### Step 3: Push Code to GitHub

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/my-first-pipeline.git

# Add files
git add .
git commit -m "Initial commit: Simple calculator app"

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Part 3: Create Your First Workflow

### Step 1: Create Workflow Directory

```bash
# Create the .github/workflows directory
mkdir -p .github/workflows
```

### Step 2: Create Workflow File

Create `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

# Trigger: Run on every push to main branch and on pull requests
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

# Jobs to run
jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    steps:
      # Step 1: Checkout code
      - name: Checkout code
        uses: actions/checkout@v3
      
      # Step 2: Set up Python
      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      # Step 3: Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      # Step 4: Run tests
      - name: Run tests
        run: |
          python test_app.py
      
      # Step 5: Print success message
      - name: Success!
        run: echo "✅ All tests passed!"
```

### Step 3: Commit and Push Workflow

```bash
git add .github/workflows/ci.yml
git commit -m "Add CI workflow"
git push
```

---

## Part 4: Verify Workflow Execution

### Step 1: View Workflow Run

1. Go to your GitHub repository
2. Click the **"Actions"** tab
3. You should see your workflow running (yellow dot) or completed (green checkmark)

### Step 2: Inspect Logs

1. Click on the workflow run
2. Click on the "Run Tests" job
3. Expand each step to see detailed logs

**What you should see:**
```
✅ Checkout code (completed in 1s)
✅ Set up Python 3.9 (completed in 3s)
✅ Install dependencies (completed in 5s)
✅ Run tests (completed in 2s)
  All tests passed!
✅ Success! (completed in 1s)
```

---

## Part 5: Trigger a Failure (Learning Exercise)

### Step 1: Introduce a Bug

Edit `app.py` and introduce a bug:

```python
def add(a, b):
    """Add two numbers"""
    return a - b  # BUG: Should be a + b
```

### Step 2: Commit and Push

```bash
git add app.py
git commit -m "Introduce bug for testing"
git push
```

### Step 3: Observe Failure

1. Go to Actions tab
2. Watch the workflow fail (red X)
3. Click on the failed run
4. See which step failed:

```
❌ Run tests (failed)
  AssertionError: assert -1 == 5
```

### Step 4: Fix the Bug

```python
def add(a, b):
    """Add two numbers"""
    return a + b  # Fixed!
```

```bash
git add app.py
git commit -m "Fix addition bug"
git push
```

Verify the workflow passes again (green checkmark).

---

## Part 6: Understanding the Workflow Syntax

### Workflow Structure Breakdown

```yaml
name: CI Pipeline              # Human-readable name
on:                            # Event triggers
  push:
    branches: [ main ]         # Only on pushes to main
  
jobs:                          # Jobs to run
  test:                        # Job ID
    runs-on: ubuntu-latest     # Operating system
    steps:                     # Sequential steps
      - name: Step name        # Step description
        uses: action/name@v1   # Use a pre-built action
      - name: Another step
        run: echo "Command"    # Run a shell command
```

### Common Triggers

```yaml
on:
  push:                        # On every push
  pull_request:                # On PR open/update
  schedule:
    - cron: '0 0 * * *'        # Daily at midnight
  workflow_dispatch:           # Manual trigger button
```

---

## Challenges

### Challenge 1: Add a Linting Step

Add a step that checks code style using `flake8`:

1. Add `flake8` to `requirements.txt`
2. Add a new step in the workflow:

```yaml
- name: Lint code
  run: flake8 app.py test_app.py --max-line-length=100
```

### Challenge 2: Test on Multiple Python Versions

Modify the workflow to test on Python 3.8, 3.9, and 3.10:

<details>
<summary>Hint</summary>

Use a matrix strategy:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
    steps:
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
```
</details>

### Challenge 3: Add a Badge to README

Add a status badge to your README.md:

```markdown
![CI](https://github.com/YOUR_USERNAME/my-first-pipeline/workflows/CI%20Pipeline/badge.svg)
```

---

## Success Criteria

✅ Workflow file exists at `.github/workflows/ci.yml`  
✅ Workflow triggers on push to main  
✅ All steps complete successfully (green checkmarks)  
✅ You can view detailed logs for each step  
✅ You understand how to debug failed workflows  

---

## Key Learnings

- **GitHub Actions workflows are defined in YAML** - Stored in `.github/workflows/`
- **Workflows consist of jobs, jobs consist of steps** - Steps run sequentially
- **Actions are reusable components** - `actions/checkout`, `actions/setup-python`
- **Logs are your best friend** - Always check logs when debugging failures
- **Fast feedback is powerful** - You know within minutes if your code works

---

## Troubleshooting

### Issue 1: Workflow doesn't trigger

**Symptoms:** No workflow run appears in Actions tab

**Solutions:**
- Ensure workflow file is in `.github/workflows/` directory
- Check YAML syntax (use a YAML validator)
- Verify you pushed to the correct branch (`main`)

### Issue 2: "Permission denied" errors

**Symptoms:** Workflow fails with permission errors

**Solutions:**
- Check repository settings → Actions → Workflow permissions
- Ensure "Read and write permissions" is enabled

### Issue 3: Tests pass locally but fail in CI

**Symptoms:** `python test_app.py` works on your laptop but fails in GitHub Actions

**Solutions:**
- Check Python version matches (`python --version`)
- Ensure all dependencies are in `requirements.txt`
- Look for environment-specific issues (file paths, etc.)

---

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

---

## Next Steps

- **Lab 06.3: Automated Testing** - Add more comprehensive tests
- **Lab 06.4: Docker Integration** - Build and push Docker images in CI

---

**Estimated Time:** 45-60 minutes  
**Difficulty:** Beginner  
**Type:** Hands-On
