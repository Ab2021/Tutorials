# Lab 1.3: Understanding the DevOps Lifecycle

## 🎯 Objective

Master the DevOps Lifecycle (the "Infinity Loop") by simulating a complete pass through all 8 phases: Plan, Code, Build, Test, Release, Deploy, Operate, and Monitor. You will manually perform the actions that automation handles in mature environments to understand the underlying flow.

## 📋 Prerequisites

- Completed Lab 1.1 and 1.2
- Basic command line familiarity
- Text editor installed
- Python installed (for the sample application)

## 🧰 Required Tools

- **Git**: For version control (Code phase)
- **Python**: For application runtime (Build/Run phase)
- **Text Editor**: For planning and coding
- **Terminal**: For execution

## 📚 Background

### The DevOps Infinity Loop

The DevOps lifecycle is often visualized as an infinity loop, representing the continuous nature of software delivery and improvement. It bridges the gap between Development (Dev) and Operations (Ops).

**The 8 Phases:**

1.  **Plan** (Dev): Define requirements, user stories, and tasks.
2.  **Code** (Dev): Write software and infrastructure code.
3.  **Build** (Dev): Compile code, resolve dependencies, package artifacts.
4.  **Test** (Dev/QA): Verify functionality, security, and performance.
5.  **Release** (Dev/Ops): Manage versions, changelogs, and release artifacts.
6.  **Deploy** (Ops): Provision infrastructure and deploy application.
7.  **Operate** (Ops): Manage configuration, scaling, and availability.
8.  **Monitor** (Ops): Observe performance, logs, and user feedback.

---

## 📖 Theory Review

### Left Side: Development (The "Dev")

-   **Plan**: Agile methodologies (Scrum/Kanban). Tools: Jira, Trello, GitHub Projects.
-   **Code**: Version control and code review. Tools: Git, VS Code, GitHub/GitLab.
-   **Build**: Continuous Integration. Tools: Jenkins, GitHub Actions, Maven, npm.
-   **Test**: Automated testing pyramids (Unit, Integration, E2E). Tools: Selenium, JUnit, PyTest.

### Right Side: Operations (The "Ops")

-   **Release**: Artifact management and versioning. Tools: Docker Hub, Nexus, Artifactory.
-   **Deploy**: Infrastructure as Code and orchestration. Tools: Terraform, Kubernetes, Ansible.
-   **Operate**: Configuration management and scaling. Tools: Ansible, Chef, Kubernetes HPA.
-   **Monitor**: Observability and alerting. Tools: Prometheus, Grafana, ELK Stack, Splunk.

### The "Continuous" Aspect

-   **Continuous Integration (CI)**: Plan → Code → Build → Test
-   **Continuous Delivery (CD)**: Release → Deploy
-   **Continuous Feedback**: Operate → Monitor → Plan

---

## 🔨 Hands-On Implementation

In this lab, we will simulate the lifecycle of a simple "Weather Dashboard" feature.

### Phase 1: Plan 📅

**Objective:** Define the work to be done.

1.  **Create a Project Directory:**
    ```bash
    mkdir devops-lifecycle-lab
    cd devops-lifecycle-lab
    ```

2.  **Create a Planning Document:**
    Create a file named `PLAN.md`.

    ```markdown
    # Project Plan: Weather Dashboard Feature

    ## User Story
    As a user, I want to see the current temperature so that I can decide what to wear.

    ## Tasks
    - [ ] Create a Python script to display temperature.
    - [ ] Hardcode a mock temperature for version 1.0.
    - [ ] Add unit tests to verify the display format.
    - [ ] Package the script for distribution.
    ```

3.  **Review:** This document serves as your "Jira ticket" or backlog item.

### Phase 2: Code 💻

**Objective:** Implement the feature using Version Control.

1.  **Initialize Git:**
    ```bash
    git init
    ```

2.  **Write the Application Code:**
    Create `weather_app.py`.

    ```python
    class WeatherApp:
        def get_temperature(self):
            # Mock database/API call
            return 25

        def display_weather(self):
            temp = self.get_temperature()
            return f"Current temperature is {temp}°C"

    if __name__ == "__main__":
        app = WeatherApp()
        print(app.display_weather())
    ```

3.  **Commit the Code:**
    ```bash
    git add .
    git commit -m "Feat: Implement basic weather display"
    ```

### Phase 3: Build 🏗️

**Objective:** Prepare the application for testing and distribution.
*In Python, "building" often means resolving dependencies or creating a package. We will simulate this by creating a requirements file and a "binary" (executable).*

1.  **Define Dependencies:**
    Create `requirements.txt`.
    ```text
    # No external dependencies for v1.0
    ```

2.  **Simulate Build Artifact:**
    We will create a zip file to represent our "build artifact".
    ```bash
    # Linux/Mac
    zip weather-app-v1.0.zip weather_app.py requirements.txt

    # Windows (PowerShell)
    Compress-Archive -Path weather_app.py, requirements.txt -DestinationPath weather-app-v1.0.zip
    ```

3.  **Verify Artifact:**
    Check that the zip file exists. This is what would be passed to the Test phase.

### Phase 4: Test 🧪

**Objective:** Verify the build artifact works as expected.

1.  **Write a Test Script:**
    Create `test_weather.py`.

    ```python
    import unittest
    from weather_app import WeatherApp

    class TestWeatherApp(unittest.TestCase):
        def test_display_format(self):
            app = WeatherApp()
            output = app.display_weather()
            self.assertIn("Current temperature is", output)
            self.assertIn("°C", output)

    if __name__ == '__main__':
        unittest.main()
    ```

2.  **Run the Test:**
    ```bash
    python3 test_weather.py
    ```
    *Expected Output:* `OK`

3.  **Simulate QA Sign-off:**
    Update `PLAN.md` to mark tasks as complete.

### Phase 5: Release 📦

**Objective:** Version and publish the artifact.

1.  **Tag the Version:**
    ```bash
    git tag v1.0.0
    ```

2.  **Create Release Notes:**
    Create `RELEASE_NOTES.md`.

    ```markdown
    # Release v1.0.0

    ## Features
    - Basic temperature display (Mock data)

    ## Hash
    (Run 'git rev-parse HEAD' and paste here)
    ```

3.  **"Publish" the Artifact:**
    Move the zip file to a simulated "Release Repository" folder.
    ```bash
    mkdir releases
    mv weather-app-v1.0.zip releases/
    ```

### Phase 6: Deploy 🚀

**Objective:** Install the application into a "Production" environment.

1.  **Create Production Directory:**
    ```bash
    mkdir production_env
    ```

2.  **Deploy the Artifact:**
    Unzip the release into the production environment.
    ```bash
    # Linux/Mac
    unzip releases/weather-app-v1.0.zip -d production_env/

    # Windows (PowerShell)
    Expand-Archive -Path releases/weather-app-v1.0.zip -DestinationPath production_env/
    ```

3.  **Configure Environment:**
    Create a `config.ini` in `production_env/` (Simulating environment-specific config).
    ```ini
    [DEFAULT]
    Environment=Production
    Debug=False
    ```

### Phase 7: Operate ⚙️

**Objective:** Run the application and ensure it stays running.

1.  **Run the Application:**
    Execute the code from the production environment.
    ```bash
    cd production_env
    python3 weather_app.py
    ```
    *Expected Output:* `Current temperature is 25°C`

2.  **Simulate Operational Task (Scaling):**
    Imagine we need to run two instances. Open a second terminal and run it again. (In a real scenario, this would be `kubectl scale` or increasing ASG size).

### Phase 8: Monitor 📊

**Objective:** Observe the application's behavior.

1.  **Simulate Logging:**
    Modify `weather_app.py` in the `production_env` folder (Hotfix simulation) to log to a file.
    *Note: In reality, you should go back to Code phase, but for this lab, we are simulating an urgent Ops change.*

    Update `production_env/weather_app.py`:
    ```python
    import datetime

    class WeatherApp:
        def get_temperature(self):
            return 25

        def display_weather(self):
            temp = self.get_temperature()
            msg = f"Current temperature is {temp}°C"
            # Log the event
            with open("app.log", "a") as f:
                f.write(f"{datetime.datetime.now()} - INFO - {msg}\n")
            return msg

    if __name__ == "__main__":
        app = WeatherApp()
        print(app.display_weather())
    ```

2.  **Generate Traffic:**
    Run the app multiple times.
    ```bash
    python3 weather_app.py
    python3 weather_app.py
    ```

3.  **Check Logs:**
    ```bash
    cat app.log
    # Windows: Get-Content app.log
    ```

---

## 🔄 The Feedback Loop

You discovered in the **Monitor** phase that logging was missing. This feeds back into the **Plan** phase for v1.1.

1.  **Update Plan:** Add "Implement structured logging" to `PLAN.md`.
2.  **Code:** Commit the logging change to the repository (not just the production file!).
3.  **Build/Test/Release/Deploy:** Go through the cycle again.

---

## 🎯 Challenges

### Challenge 1: The Broken Build (Difficulty: ⭐⭐)

**Scenario:**
Introduce a syntax error in `weather_app.py`.

**Task:**
1.  Attempt to run the "Build" phase (create zip).
2.  Attempt to run the "Test" phase.
3.  Document at which phase the error should ideally be caught.

### Challenge 2: Automate the Cycle (Difficulty: ⭐⭐⭐⭐)

**Scenario:**
Manual steps are prone to error. Create a simple shell script (or PowerShell script) named `pipeline.sh` (or `pipeline.ps1`) that automates Phases 3, 4, and 6.

**Requirements:**
-   Script should run tests.
-   If tests pass, create the zip artifact.
-   If zip created, deploy to `production_env`.
-   Stop immediately if any step fails.

**Deliverable:**
-   `pipeline.sh` / `pipeline.ps1` code.

---

## 💡 Solution

<details>
<summary>Click to reveal Challenge 2 Solution (Bash)</summary>

```bash
#!/bin/bash
# Simple DevOps Pipeline Script

echo "🚀 Starting Pipeline..."

# Phase 4: Test
echo "🧪 Running Tests..."
python3 test_weather.py
if [ $? -ne 0 ]; then
    echo "❌ Tests Failed! Stopping pipeline."
    exit 1
fi
echo "✅ Tests Passed."

# Phase 3: Build
echo "🏗️ Building Artifact..."
zip weather-app-v1.1.zip weather_app.py requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Build Failed!"
    exit 1
fi
echo "✅ Build Complete."

# Phase 6: Deploy
echo "🚀 Deploying to Production..."
mkdir -p production_env
unzip -o weather-app-v1.1.zip -d production_env/
echo "✅ Deployed Successfully."

echo "🎉 Pipeline Finished!"
```
</details>

<details>
<summary>Click to reveal Challenge 2 Solution (PowerShell)</summary>

```powershell
# Simple DevOps Pipeline Script

Write-Host "🚀 Starting Pipeline..." -ForegroundColor Cyan

# Phase 4: Test
Write-Host "🧪 Running Tests..." -ForegroundColor Yellow
python test_weather.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Tests Failed! Stopping pipeline." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Tests Passed." -ForegroundColor Green

# Phase 3: Build
Write-Host "🏗️ Building Artifact..." -ForegroundColor Yellow
$artifactName = "weather-app-v1.1.zip"
Compress-Archive -Path weather_app.py, requirements.txt -DestinationPath $artifactName -Force
if (-not (Test-Path $artifactName)) {
    Write-Host "❌ Build Failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build Complete." -ForegroundColor Green

# Phase 6: Deploy
Write-Host "🚀 Deploying to Production..." -ForegroundColor Yellow
if (-not (Test-Path "production_env")) { New-Item -ItemType Directory -Force -Path "production_env" | Out-Null }
Expand-Archive -Path $artifactName -DestinationPath "production_env" -Force
Write-Host "✅ Deployed Successfully." -ForegroundColor Green

Write-Host "🎉 Pipeline Finished!" -ForegroundColor Cyan
```
</details>

---

## 🔑 Key Takeaways

1.  **Interconnectedness**: Output of one phase is input for the next.
2.  **Feedback**: Monitoring data informs future planning.
3.  **Automation**: Manual lifecycles are slow and risky; scripts (Challenge 2) are the first step toward CI/CD.
4.  **Artifacts**: Build once, deploy anywhere (we moved the same zip file).

---

## ⏭️ Next Steps

Now that you understand the lifecycle, we need to set up the actual tools used in the industry.

Proceed to **Lab 1.4: Setting Up the DevOps Toolchain**.
