# Lab 6.2: GitHub Actions Basics

## 🎯 Objective

Write your first automated pipeline. You will create a GitHub Actions workflow that prints "Hello World" and runs a simple script whenever you push code.

## 📋 Prerequisites

-   GitHub Account.
-   A GitHub Repository.

## 📚 Background

### Anatomy of a Workflow
-   **File**: `.github/workflows/main.yml`
-   **Triggers (`on`)**: When to run? (Push, Pull Request, Schedule).
-   **Jobs**: A set of steps running on a runner.
-   **Steps**: Individual commands (`run`) or actions (`uses`).
-   **Runner**: The server running the job (`ubuntu-latest`).

---

## 🔨 Hands-On Implementation

### Part 1: Create the Workflow 📄

1.  **In your local repo:**
    ```bash
    mkdir -p .github/workflows
    nano .github/workflows/hello.yml
    ```

2.  **Add Content:**
    ```yaml
    name: First Workflow

    on: [push]

    jobs:
      say-hello:
        runs-on: ubuntu-latest
        steps:
          - name: Checkout Code
            uses: actions/checkout@v2

          - name: Print Greeting
            run: echo "Hello, GitHub Actions!"

          - name: Check OS
            run: uname -a
    ```

3.  **Commit and Push:**
    ```bash
    git add .
    git commit -m "Add workflow"
    git push
    ```

### Part 2: Verify Execution ✅

1.  **Go to GitHub:**
    Click the **Actions** tab in your repository.

2.  **Check Status:**
    You should see "First Workflow" listed.
    Click it -> Click "say-hello" job.

3.  **View Logs:**
    Expand "Print Greeting". You should see `Hello, GitHub Actions!`.

### Part 3: Adding a Script 📜

1.  **Create `script.sh`:**
    ```bash
    #!/bin/bash
    echo "Running a script from the repo!"
    ls -la
    ```
    Make executable: `chmod +x script.sh`

2.  **Update Workflow:**
    Add a step:
    ```yaml
          - name: Run Script
            run: ./script.sh
    ```

3.  **Push:**
    Watch the Actions tab again.

---

## 🎯 Challenges

### Challenge 1: Multi-OS Matrix (Difficulty: ⭐⭐⭐)

**Task:**
Run the job on Ubuntu, Windows, and MacOS simultaneously.
*Hint: Use `strategy: matrix`.*

### Challenge 2: Conditional Step (Difficulty: ⭐⭐)

**Task:**
Add a step that ONLY runs if the branch is `main`.
*Hint: `if: github.ref == 'refs/heads/main'`*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
```

**Challenge 2:**
```yaml
- name: Deploy
  if: github.ref == 'refs/heads/main'
  run: echo "Deploying..."
```
</details>

---

## 🔑 Key Takeaways

1.  **YAML**: Indentation matters!
2.  **Marketplace**: You don't have to write everything. Use `actions/checkout`, `actions/setup-python`, etc.
3.  **Free Tier**: GitHub gives 2000 free minutes/month. Use them wisely.

---

## ⏭️ Next Steps

We can print text. Now let's test code.

Proceed to **Lab 6.3: Automated Testing with CI**.
