# Lab 6.1: Introduction to CI/CD Concepts

## 🎯 Objective

Understand the "Pipeline". You will simulate a CI/CD pipeline manually to understand the steps before automating them.

## 📋 Prerequisites

-   None (Conceptual Lab).

## 📚 Background

### CI (Continuous Integration)
**Goal**: Merge code frequently (daily).
**Steps**:
1.  **Checkout**: Get code.
2.  **Build**: Compile/Install dependencies.
3.  **Test**: Run unit tests.
4.  **Merge**: If pass, merge to main.

### CD (Continuous Delivery/Deployment)
**Goal**: Release code frequently.
**Steps**:
1.  **Artifact**: Package the app (Docker Image / Zip).
2.  **Deploy (Staging)**: Put it on a test server.
3.  **Verify**: Integration tests.
4.  **Deploy (Prod)**: Put it on the live server.
    -   *Delivery*: Manual approval button.
    -   *Deployment*: Automatic.

---

## 🔨 Hands-On Implementation

### Part 1: The Manual Pipeline Simulation 🏃‍♂️

**Scenario:** You are the "Build Server".

1.  **Step 1: Checkout**
    Create a folder `pipeline_sim`.
    Create `app.py` (The code).
    ```python
    print("Hello World")
    ```

2.  **Step 2: Test**
    Run the test manually.
    ```bash
    python3 app.py
    ```
    *Result:* It prints "Hello World". **PASS**.

3.  **Step 3: Build (Package)**
    Create the artifact.
    ```bash
    tar -czf app-v1.tar.gz app.py
    ```

4.  **Step 4: Deploy**
    Move artifact to "Production" folder.
    ```bash
    mkdir production
    cp app-v1.tar.gz production/
    cd production
    tar -xzf app-v1.tar.gz
    ```

### Part 2: The Failure Scenario 💥

1.  **Dev breaks code:**
    Modify `app.py` to have a syntax error.
    ```python
    print("Hello World"  # Missing parenthesis
    ```

2.  **Run Test:**
    ```bash
    python3 app.py
    ```
    *Result:* `SyntaxError`. **FAIL**.

3.  **Action:**
    **STOP THE LINE.** Do not package. Do not deploy. Email the developer.

---

## 🎯 Challenges

### Challenge 1: The Cost of Manual (Difficulty: ⭐)

**Task:**
Time yourself doing Part 1. (e.g., 30 seconds).
Multiply by 10 developers committing 5 times a day.
(30s * 50 = 25 minutes/day lost).
Now imagine the tests take 10 minutes, not 1 second.

---

## 🔑 Key Takeaways

1.  **Consistency**: Humans forget steps. Scripts don't.
2.  **Feedback Loop**: CI gives developers instant feedback ("You broke it") instead of waiting for QA week.
3.  **Artifacts**: Build once, deploy everywhere. Never re-build on production.

---

## ⏭️ Next Steps

Let's replace "You" with a robot.

Proceed to **Lab 6.2: GitHub Actions Basics**.
