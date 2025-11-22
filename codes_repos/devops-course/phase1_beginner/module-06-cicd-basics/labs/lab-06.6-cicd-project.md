# Lab 6.6: CI/CD Capstone Project

## 🎯 Objective

Build the **Ultimate Pipeline**. You will create a complete, end-to-end CI/CD workflow for a Python Flask application.
**Flow**: Code -> Lint -> Test -> Build -> Push -> Deploy.

## 📋 Prerequisites

-   Completed Module 6.
-   GitHub Repo.
-   Docker Hub Account.

## 📚 Background

### The Scenario
You are the DevOps Engineer for a startup. The developers want to just "git push" and go home. Your job is to make that safe and possible.

---

## 🔨 Hands-On Implementation

### Step 1: The Application 🐍

1.  **Structure:**
    ```text
    /
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    ├── test_app.py
    └── .github/workflows/pipeline.yml
    ```

2.  **`app.py`:**
    ```python
    from flask import Flask
    app = Flask(__name__)

    @app.route('/')
    def hello():
        return "Hello DevOps World!"

    if __name__ == "__main__":
        app.run(host='0.0.0.0')
    ```

3.  **`test_app.py`:**
    ```python
    import unittest
    from app import app

    class TestApp(unittest.TestCase):
        def test_home(self):
            tester = app.test_client(self)
            response = tester.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, b"Hello DevOps World!")
    ```

4.  **`Dockerfile`:**
    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /app
    COPY . .
    RUN pip install flask
    CMD ["python", "app.py"]
    ```

### Step 2: The Pipeline 🛤️

Create `.github/workflows/pipeline.yml`.

```yaml
name: Ultimate CI/CD

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  # Job 1: Quality Control
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install Deps
        run: pip install flask pylint
      - name: Lint
        run: pylint app.py --disable=C0114,C0115,C0116 # Ignore docstring warnings

  # Job 2: Tests
  test:
    needs: quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install Deps
        run: pip install flask
      - name: Run Tests
        run: python -m unittest test_app.py

  # Job 3: Build & Push (Only on Push to Main)
  build-push:
    needs: test
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Login to Docker Hub
        uses: docker/login-action@v1
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - name: Build and Push
        uses: docker/build-push-action@v2
        with:
          context: .
          push: true
          tags: ${{ secrets.DOCKER_USERNAME }}/capstone:latest

  # Job 4: Deploy (Simulation)
  deploy:
    needs: build-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Prod
        run: echo "🚀 Deploying to Production Server... Done!"
```

### Step 3: Execution 🎬

1.  **Commit & Push:**
    ```bash
    git add .
    git commit -m "Init Capstone"
    git push origin main
    ```

2.  **Verify:**
    Watch the pipeline visualization.
    `Quality` -> `Test` -> `Build` -> `Deploy`.

### Step 4: The PR Check 🛡️

1.  **Create Branch:**
    ```bash
    git checkout -b feature/bad-code
    ```
2.  **Break Test:**
    Change "Hello DevOps World!" to "Bye".
3.  **Push & PR:**
    Open a Pull Request.
4.  **Observe:**
    -   `Quality`: Pass.
    -   `Test`: **Fail**.
    -   `Build`: **Skipped** (because Test failed).
    -   `Deploy`: **Skipped**.
    -   **Result**: PR cannot be merged (if branch protection is on).

---

## 🎯 Challenges

### Challenge 1: Slack Notification (Difficulty: ⭐⭐⭐)

**Task:**
Add a step at the end of the pipeline to send a message to a Slack channel (or Discord) if the build fails.
*Hint: Use a Marketplace Action like `rtCamp/action-slack-notify`.*

### Challenge 2: Artifact Upload (Difficulty: ⭐⭐)

**Task:**
If the tests fail, upload the log file as an Artifact so you can download and inspect it later.
*Hint: `uses: actions/upload-artifact@v2`.*

---

## 🔑 Key Takeaways

1.  **Dependencies**: `needs: [job1, job2]` defines the order.
2.  **Conditionals**: `if: github.event_name == 'push'` ensures we don't deploy from a PR.
3.  **Confidence**: When the pipeline is green, you *know* the code is good.

---

## ⏭️ Next Steps

**Congratulations!** You have completed Phase 1 (Beginner).
You now possess the core skills: Linux, Python, Git, Networking, Docker, and CI/CD.

**Phase 2 (Intermediate)** awaits. We will tackle Infrastructure as Code (Terraform) and Orchestration (Kubernetes).

Proceed to **Phase 2: Module 7 - Infrastructure as Code**.
