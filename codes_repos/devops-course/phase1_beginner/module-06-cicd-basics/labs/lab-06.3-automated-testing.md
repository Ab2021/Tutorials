# Lab 6.3: Automated Testing with CI

## 🎯 Objective

Stop bugs before they reach production. You will configure a CI pipeline that runs Unit Tests automatically. If the tests fail, the pipeline turns red, and you (or your boss) know immediately.

## 📋 Prerequisites

-   Completed Lab 6.2.
-   Basic Python knowledge.

## 📚 Background

### The Test Pyramid
1.  **Unit Tests**: Fast, cheap. Test individual functions. (We are here).
2.  **Integration Tests**: Test how modules talk to each other.
3.  **E2E Tests**: Slow, expensive. Test the whole app (Selenium/Cypress).

### CI Workflow
`Push` -> `Install Deps` -> `Run Tests` -> `Pass/Fail`.

---

## 🔨 Hands-On Implementation

### Part 1: The Application & Tests 🐍

1.  **Create `calc.py`:**
    ```python
    def add(x, y):
        return x + y

    def subtract(x, y):
        return x - y
    ```

2.  **Create `test_calc.py`:**
    ```python
    import unittest
    import calc

    class TestCalc(unittest.TestCase):
        def test_add(self):
            self.assertEqual(calc.add(10, 5), 15)

        def test_subtract(self):
            self.assertEqual(calc.subtract(10, 5), 5)

    if __name__ == '__main__':
        unittest.main()
    ```

3.  **Test Locally:**
    ```bash
    python3 test_calc.py
    ```
    *Result:* `OK`.

### Part 2: The CI Pipeline ⚙️

1.  **Create `.github/workflows/test.yml`:**
    ```yaml
    name: Python Tests

    on: [push]

    jobs:
      build-and-test:
        runs-on: ubuntu-latest

        steps:
          - name: Checkout Code
            uses: actions/checkout@v2

          - name: Set up Python
            uses: actions/setup-python@v2
            with:
              python-version: '3.9'

          - name: Install Dependencies
            run: |
              python -m pip install --upgrade pip
              # pip install -r requirements.txt (If we had one)

          - name: Run Tests
            run: python test_calc.py
    ```

2.  **Push:**
    ```bash
    git add .
    git commit -m "Add tests"
    git push
    ```

3.  **Verify:**
    Check GitHub Actions. Green checkmark ✅.

### Part 3: Breaking the Build 💥

1.  **Modify `calc.py`:**
    Introduce a bug.
    ```python
    def add(x, y):
        return x - y  # Oops, copy-paste error
    ```

2.  **Push:**
    ```bash
    git commit -am "Broken code"
    git push
    ```

3.  **Verify:**
    Check GitHub Actions. Red cross ❌.
    *Log:* `AssertionError: 5 != 15`.

---

## 🎯 Challenges

### Challenge 1: Branch Protection (Difficulty: ⭐⭐⭐)

**Task:**
1.  Go to Repo Settings -> Branches -> Add Rule (`main`).
2.  Check "Require status checks to pass before merging".
3.  Select "build-and-test".
4.  Try to merge a PR with broken tests. GitHub should block the "Merge" button.

### Challenge 2: Test Coverage (Difficulty: ⭐⭐)

**Task:**
Use `pytest` and `pytest-cov` to generate a coverage report.
Update the workflow to run `pytest --cov=./`.
*Goal:* See "100% coverage" in the CI logs.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
(UI based task). This is a critical DevOps control. It prevents "Cowboy Coding" on main.

**Challenge 2:**
Workflow update:
```yaml
- name: Install Deps
  run: pip install pytest pytest-cov

- name: Run Tests
  run: pytest --cov=./
```
</details>

---

## 🔑 Key Takeaways

1.  **Fail Fast**: It's better to fail in CI (private) than in Production (public).
2.  **Status Checks**: Use them to enforce quality gates on Pull Requests.
3.  **Reproducibility**: CI runs on a clean machine. If it works on your machine but fails in CI, you missed a dependency.

---

## ⏭️ Next Steps

We tested the code. Now let's package it.

Proceed to **Lab 6.4: Docker Integration**.
