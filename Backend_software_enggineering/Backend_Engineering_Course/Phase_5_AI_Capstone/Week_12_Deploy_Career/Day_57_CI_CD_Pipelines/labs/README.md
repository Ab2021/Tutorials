# Lab: Day 57 - GitHub Actions

## Goal
Build a CI pipeline.

## Prerequisites
- A GitHub repository.

## Step 1: The Code (`test_app.py`)
```python
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
```

## Step 2: The Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.9"

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest

    - name: Run Tests
      run: |
        pytest
```

## Step 3: Run It
1.  Commit these files.
2.  Push to GitHub.
3.  Go to "Actions" tab in your repo.
4.  See the "CI Pipeline" running.

## Challenge
Add a **Linting Step**.
1.  `pip install flake8`.
2.  Add a step `run: flake8 .` before testing.
3.  Make the code fail linting (e.g., unused import).
4.  Push and watch CI fail.
