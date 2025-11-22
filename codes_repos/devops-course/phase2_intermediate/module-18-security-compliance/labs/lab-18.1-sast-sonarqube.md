# Lab 18.1: SAST with SonarQube

## 🎯 Objective

Shift Left. Don't wait for a hacker to find a vulnerability in production. Find it in the code, before it's even committed. You will set up **SonarQube** to scan your Python code for bugs and security holes.

## 📋 Prerequisites

-   Docker installed.
-   Python installed.

## 📚 Background

### SAST (Static Application Security Testing)
-   Analyzes **Source Code**.
-   Finds: SQL Injection, Hardcoded Passwords, Unused Variables, Complexity.
-   Runs: In CI pipeline or locally.

---

## 🔨 Hands-On Implementation

### Part 1: Run SonarQube 🐳

1.  **Run Container:**
    ```bash
    docker run -d \
      --name sonarqube \
      -p 9000:9000 \
      sonarqube:lts-community
    ```

2.  **Login:**
    -   `http://localhost:9000`.
    -   User: `admin`, Pass: `admin` (Change it).

3.  **Create Project:**
    -   Select **Manually**.
    -   Name: `vulnerable-app`.
    -   Key: `vulnerable-app`.
    -   Click **Set Up**.

4.  **Generate Token:**
    -   Select **Locally**.
    -   Generate token named `ci-token`.
    -   **Copy this token**.

### Part 2: The Vulnerable Code 🐍

1.  **Create `app.py`:**
    ```python
    import mysql.connector

    def login(user, password):
        # VULNERABILITY: SQL Injection
        sql = "SELECT * FROM users WHERE user = '" + user + "' AND password = '" + password + "'"
        cursor.execute(sql)

    def connect():
        # VULNERABILITY: Hardcoded Password
        mydb = mysql.connector.connect(
          host="localhost",
          user="root",
          password="password123"
        )
    ```

### Part 3: The Scan 🔍

1.  **Install Scanner:**
    Download `sonar-scanner-cli` or use Docker. We will use Docker.

2.  **Run Scanner:**
    ```bash
    docker run \
        --rm \
        -e SONAR_HOST_URL="http://host.docker.internal:9000" \
        -e SONAR_SCANNER_OPTS="-Dsonar.projectKey=vulnerable-app" \
        -e SONAR_TOKEN="YOUR_TOKEN_HERE" \
        -v "$(pwd):/usr/src" \
        sonarsource/sonar-scanner-cli
    ```

### Part 4: Analyze Results 📊

1.  **Check Dashboard:**
    Go to `http://localhost:9000/dashboard?id=vulnerable-app`.

2.  **Findings:**
    -   **Security Hotspot**: "Hardcoded credentials".
    -   **Vulnerability**: "SQL Injection".
    -   **Code Smell**: "Unused import".

---

## 🎯 Challenges

### Challenge 1: Quality Gate (Difficulty: ⭐⭐)

**Task:**
Configure a **Quality Gate** in SonarQube.
Rule: "If Security Rating is worse than A, fail the gate."
Re-run the scan.
*Goal:* In a CI pipeline, this would block the Pull Request.

### Challenge 2: Fix the Code (Difficulty: ⭐⭐)

**Task:**
Fix the SQL Injection using parameterized queries.
Fix the hardcoded password using Environment Variables.
Re-run the scan.
*Result:* 0 Vulnerabilities.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
```python
import os
sql = "SELECT * FROM users WHERE user = %s AND password = %s"
cursor.execute(sql, (user, password))
password = os.environ.get("DB_PASSWORD")
```
</details>

---

## 🔑 Key Takeaways

1.  **Continuous Inspection**: Scan every commit.
2.  **Technical Debt**: SonarQube estimates how many minutes/hours it will take to fix the code.
3.  **False Positives**: SAST tools aren't perfect. You can mark issues as "Won't Fix" or "False Positive" in the UI.

---

## ⏭️ Next Steps

We secured the code. Now let's secure the artifact.

Proceed to **Lab 18.2: Container Signing with Cosign**.
