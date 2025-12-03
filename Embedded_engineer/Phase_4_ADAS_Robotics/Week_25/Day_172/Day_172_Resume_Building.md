# Day 172: Resume & Portfolio Building
## Phase 4: ADAS & Robotics Systems | Week 25: Final Assessment & Career

---

> **📝 Day 172 Focus:**
> You have the skills. Now you need to sell them. A generic software resume won't work for Robotics. You need to highlight **C++, ROS 2, and Real-Time Systems**. Today, we build your **Career Portfolio**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Structure** a Robotics-focused Resume (Skills, Projects, Experience).
2.  **Optimize** your GitHub Profile (Pinned Repos, READMEs, GIFs).
3.  **Write** a compelling Project Description (STAR method).
4.  **Tailor** your application for specific roles (Perception vs Control).
5.  **Leverage** LinkedIn for networking in the AV industry.

---

## 📚 The Robotics Resume

### 🔹 1. The Skills Section (Top Heavy)

Recruiters scan this in 6 seconds. Group by category.

*   **Languages:** C++ (14/17), Python, Bash.
*   **Robotics:** ROS 2 (Humble), TF2, URDF, Gazebo, CARLA.
*   **Algorithms:** SLAM (Gmapping), Planning (A*, RRT), Control (PID, MPC), Perception (YOLO, OpenCV).
*   **Tools:** Docker, Git, CMake, Jenkins, Linux (Ubuntu).
*   **Hardware:** Raspberry Pi, NVIDIA Jetson, Lidar (Velodyne), Camera.

### 🔹 2. The Projects Section (The Meat)

Don't just list the name. Explain the **Impact**.

*   **Autonomous Valet Parking (Capstone):**
    *   *Designed* a Level 4 parking system in ROS 2/CARLA.
    *   *Implemented* Hybrid A* planner and MPC controller for precise maneuvering (< 5cm error).
    *   *Fused* Fisheye Camera and Ultrasonic data for 360° obstacle detection.
    *   *Validated* system with 100+ regression tests using Jenkins.

*   **Lidar SLAM Implementation:**
    *   *Built* a Graph-based SLAM system from scratch in C++.
    *   *Optimized* loop closure detection using ICP, reducing drift by 40%.

### 🔹 3. Experience

*   Focus on **Engineering** tasks. "Built", "Optimized", "Debugged", "Deployed".
*   Quantify results. "Reduced latency by 20ms". "Increased coverage to 95%".

---

## 💻 Implementation: GitHub Makeover

**Scenario:**
-   Your GitHub is your portfolio.
-   An empty repo with just code is useless. You need a `README.md` that sells the project.

### 🛠️ Setup
Go to your `week24_capstone` repo.

### 👨‍💻 Code: The Perfect README.md

```markdown
# Autonomous Valet Parking (AVP) System 🚗🅿️

[![ROS 2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Build Status](https://github.com/username/avp/actions/workflows/test.yml/badge.svg)](actions)

A complete Level 4 Autonomous Valet Parking stack for ROS 2, simulated in CARLA. Capable of mapping, planning, and executing complex parking maneuvers in GPS-denied environments.

## 🎥 Demo
![AVP Demo GIF](docs/demo.gif)
*(Click to watch full video on YouTube)*

## 🚀 Features
- **Hybrid A* Planner:** Generates kinematically feasible paths with gear shifts (Drive/Reverse).
- **MPC Controller:** Custom Model Predictive Control for sub-5cm stopping accuracy.
- **Perception:** 4-Camera Surround View + Ultrasonic Fusion.
- **Simulation:** Full integration with CARLA Town04.

## 🛠️ Tech Stack
- **Language:** C++17, Python 3.10
- **Middleware:** ROS 2 Humble
- **Libraries:** OpenCV, PCL, Eigen, CVXPY, Ceres Solver

## 📦 Installation
```bash
git clone https://github.com/username/avp_stack.git
colcon build --symlink-install
source install/setup.bash
```

## 🏃‍♂️ Usage
```bash
ros2 launch avp_stack system.launch.py
```

## 📐 Architecture
![Architecture Diagram](docs/architecture.png)

## 🤝 Contributing
Pull requests are welcome. Please read `CONTRIBUTING.md`.
```

---

## 🔬 Lab Exercise: Portfolio Audit

### Lab Objectives
1.  **Google Yourself:** What comes up?
2.  **Check GitHub:**
    -   Do you have a profile picture?
    -   Are your top 3 repos pinned?
    -   Do they have GIFs? (People love GIFs).
3.  **Check LinkedIn:**
    -   Headline: "Robotics Software Engineer | ROS 2 | C++ | SLAM".
    -   About: Short summary of your passion and skills.
    -   Featured: Link your AVP Demo Video.

---

## 🐞 Debugging & Troubleshooting

### Common Resume Mistakes

#### 1. The "Jack of All Trades"
**Mistake:** Listing HTML, CSS, React, Java, C++, Python, SQL, AWS, Azure...
**Fix:** Tailor it. If applying for Robotics, hide the Web Dev stuff. It looks unfocused.

#### 2. The "Wall of Text"
**Mistake:** 5-line paragraphs.
**Fix:** Bullet points. Max 2 lines per bullet.

#### 3. Broken Links
**Mistake:** GitHub link 404s.
**Fix:** Double check every link.

---

## ⚡ Optimization & Best Practices

### 1. The ATS (Applicant Tracking System)
-   Robots read your resume before humans do.
-   Keywords matter. "ROS", "C++", "SLAM", "Lidar".
-   Don't use fancy graphics or columns that confuse the parser. Keep it simple (Single column).

### 2. Cover Letters
-   Don't rewrite your resume.
-   Tell a story. "I've been following Waymo's progress in San Francisco, and I built a similar perception stack in my capstone..."
-   Show passion for *their* mission.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Should I include my GPA?
    *   **A:** Only if it's > 3.5 and you are a fresh grad. Otherwise, experience > grades.
2.  **Q:** How many pages?
    *   **A:** One page. Two if you have 10+ years of experience or a PhD.
3.  **Q:** What if I don't have "Real" experience?
    *   **A:** Your Capstone Project *is* experience. Treat it like a job. "Lead Engineer for AVP Project".

### Challenge Task
**Task:** The Elevator Pitch.
1.  Write a 30-second intro about yourself.
2.  "Hi, I'm [Name]. I'm a Robotics Engineer specializing in Navigation. I recently built an autonomous parking system using ROS 2 and MPC, and I'm looking for roles where I can work on production L4 stacks."

---

## 📚 Further Reading & References
-   [Awesome Robotics Jobs](https://github.com/vchrombie/awesome-robotics-jobs)
-   [Resume Worded (ATS Checker)](https://resumeworded.com/)

---

**Day 172 Complete** | Phase 4: ADAS & Robotics Systems | Week 25: Final Assessment & Career
