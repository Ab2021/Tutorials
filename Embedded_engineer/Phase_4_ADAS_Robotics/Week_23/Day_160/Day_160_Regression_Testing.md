# Day 160: Regression Testing
## Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation

---

> **📝 Day 160 Focus:**
> You fixed the "Cut-in" bug. Great! But did you break the "Parking" logic? **Regression Testing** is the practice of running *all* previous tests every time you make a change. It prevents the "One step forward, two steps back" problem.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** Regression Testing and its importance in Safety-Critical Systems.
2.  **Organize** a Test Suite (Smoke, Sanity, Full Regression).
3.  **Configure** a Jenkins Pipeline to execute tests automatically.
4.  **Analyze** Test Reports (JUnit XML) and track trends.
5.  **Implement** "Nightly Builds" for long-running simulations.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 144:** CI/CD (GitHub Actions).
-   **Day 159:** OpenSCENARIO.

### Hardware Requirements
-   **Build Server:** A powerful PC/Server to run simulations.

### Software Stack
-   **Jenkins:** Automation Server.
-   **Allure:** Reporting tool.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Testing Pyramid

1.  **Unit Tests (Thousands):** Fast (ms). Run on every commit. (Day 144).
2.  **Integration/SIL Tests (Hundreds):** Medium (seconds). Run on Merge Request. (Day 156).
3.  **System/Scenario Tests (Tens):** Slow (minutes/hours). Run Nightly. (Day 159).

### 🔹 Part 2: Test Selection Strategy

Running 10,000 scenarios takes too long.
-   **Smoke Test:** Critical path only (Start car, Drive straight). If this fails, stop.
-   **Impact Analysis:** Only run tests related to the changed module (e.g., if `planner` changed, run `planner_tests`, skip `perception_tests`).

### 🔹 Part 3: Reporting

-   **Pass/Fail:** Binary result.
-   **KPIs:** "Min Distance to Obstacle", "Max Jerk".
-   **Artifacts:** Logs, Rosbags, Videos of failures.

---

## 💻 Implementation: Jenkins Pipeline

**Scenario:**
-   We have a repo with `src` and `tests`.
-   We want Jenkins to:
    1.  Build the workspace.
    2.  Run Unit Tests.
    3.  Run a CARLA Scenario.
    4.  Publish Report.

### 🛠️ Setup
Assume Jenkins is installed (`sudo apt install jenkins`).
Create `week23_day160` and `Jenkinsfile`.

```bash
mkdir -p ~/ros2_ws/src/week23_day160
cd ~/ros2_ws/src/week23_day160
touch Jenkinsfile
```

### 👨‍💻 Code: The Jenkinsfile (Groovy)

```groovy
pipeline {
    agent any
    
    environment {
        ROS_DISTRO = 'humble'
        WS_DIR = "${WORKSPACE}/ros2_ws"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build') {
            steps {
                sh """
                source /opt/ros/${ROS_DISTRO}/setup.bash
                colcon build --symlink-install
                """
            }
        }
        
        stage('Unit Tests') {
            steps {
                sh """
                source /opt/ros/${ROS_DISTRO}/setup.bash
                colcon test --packages-select week23_day156
                colcon test-result --verbose
                """
            }
        }
        
        stage('Scenario Tests') {
            steps {
                // This requires CARLA to be running or Dockerized
                sh """
                source install/setup.bash
                python3 scenario_runner.py --openscenario src/week23_day159/cut_in.xosc --output --junit
                """
            }
            post {
                always {
                    junit 'test_results/*.xml'
                }
            }
        }
    }
    
    post {
        failure {
            mail to: 'dev@company.com',
                 subject: "Build Failed: ${env.JOB_NAME} [${env.BUILD_NUMBER}]",
                 body: "Check console output at ${env.BUILD_URL}"
        }
    }
}
```

---

## 🔬 Lab Exercise: The Nightly Run

### Lab Objectives
1.  **Mock the Pipeline:**
    Since we don't have a Jenkins server, we run the steps manually in a script `run_regression.sh`.
    ```bash
    #!/bin/bash
    set -e
    
    echo "BUILDING..."
    colcon build
    
    echo "UNIT TESTS..."
    colcon test
    
    echo "SCENARIO TESTS..."
    # Mocking the scenario runner return code
    # python3 scenario_runner.py ...
    echo "Scenario Passed"
    
    echo "GENERATING REPORT..."
    colcon test-result --all
    ```
2.  **Inject a Bug:**
    -   Modify `aeb_node.cpp` to invert the logic (`if dist > 10`).
    -   Run `run_regression.sh`.
    -   **Result:** Unit Tests pass (maybe), but Scenario Test fails (Crash).
    -   **Lesson:** Unit tests are not enough.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Flaky Scenarios
**Symptom:** Scenario fails 1 out of 10 times.
**Cause:** Simulator timing, Physics non-determinism.
**Solution:** Run the scenario 3 times. If it passes 2/3, mark as "Unstable" but don't block the build. Or fix the determinism (Sync Mode).

#### 2. Disk Space
**Symptom:** Build server crashes.
**Cause:** Rosbags and Docker images fill the disk.
**Solution:** Configure Jenkins "Discard Old Builds" (Keep last 10). Delete artifacts after 24h.

---

## ⚡ Optimization & Best Practices

### 1. Parallel Execution
-   Run scenarios in parallel.
-   Jenkins: `parallel { stage('Scenario A') { ... } stage('Scenario B') { ... } }`.
-   Requires multiple GPU nodes or lightweight simulation.

### 2. Dashboarding
-   Use **Grafana** or **Allure** to visualize pass rates over time.
-   "Are we getting better or worse?"

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is a "Smoke Test"?
    *   **A:** A quick, basic test to verify the system isn't completely broken (e.g., does it compile? does it start?).
2.  **Q:** Why JUnit XML format?
    *   **A:** It's the universal standard for test results. Jenkins, GitLab, and Allure all understand it.
3.  **Q:** What is "Continuous Deployment" (CD)?
    *   **A:** Automatically deploying the code to the car (OTA) after it passes all tests. (Risky for AVs, usually requires manual approval).

### Challenge Task
**Task:** KPI Extraction.
1.  Parse the `scenario_runner` log.
2.  Extract `min_distance` for each run.
3.  Plot a histogram.
4.  If `mean(min_distance)` decreases over time, the code is degrading.

---

## 📚 Further Reading & References
-   [Jenkins Pipeline Syntax](https://www.jenkins.io/doc/book/pipeline/syntax/)
-   [Allure Framework](https://docs.qameta.io/allure/)

---

**Day 160 Complete** | Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation
