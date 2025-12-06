# Day 209: Portfolio & Career Preparation
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 30: Capstone Project Part 2

---

> **📝 Content Creator Instructions:**
> Get Hired.
> - **Focus:** Building a Killer Portfolio. Optimizing the Resume for parsing algorithms (ATS). Mock Technical Interviews.
> - **Code:** `resume_robotics.tex` (LaTeX Resume Template) and `portfolio_layout.html`.
> - **Concept:** Self-Marketing.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Format** a robotics-focused resume (Skills: ROS 2, C++, Python, PyTorch).
2.  **Deploy** a GitHub Pages portfolio showcasing the Capstone videos and code.
3.  **Answer** common behavioral questions ("Tell me about a bug you fixed").
4.  **Prepare** for the "Whiteboard Coding" round (LeetCode for Robotics).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
- LaTeX Editor (Overleaf or local).
- Hugo/Jekyll for Portfolio.

### Prior Knowledge
- All 208 days of hard work.

---

## 📖 Theoretical Hiring Funnel

### 🔹 The 3 Stages

1.  **The Recruiter Screen:** Non-technical. Matches keywords ("ROS", "C++", "Lidar").
    *   *Strategy:* Ensure your resume hits every buzzword in the Job Description.
2.  **The Technical Screen:** 1 hour. Algorithm questions (DFS/BFS) or Domain questions (Transformation Matrices).
    *   *Strategy:* Know your Linear Algebra and C++ pointers cold.
3.  **The Onsite:** 4-5 hours. System Design (Design a Vacuum Robot) + Behavioral.
    *   *Strategy:* The STAR Method. Design for Safety and Scalability.

### 🔹 The Portfolio Effect

A video of a working robot > GPA.
*   **Repo:** Clean code, Unit Tests, CI.
*   **Readme:** Documentation.
*   **Video:** 30s summary.

---

## 💻 Implementation: The Assets

### 🛠️ Project Structure
```text
career_prep/
├── resume/
│   └── robotics_engineer.tex
├── portfolio/
│   └── index.html
└── interview_prep/
    └── common_questions.md
```

### 👨‍💻 Resume Template (`resume/robotics_engineer.tex`)

```latex
\documentclass{article}
\usepackage{titlesec}

% Section: Skills
\section{Technical Skills}
\textbf{Languages:} C++ (14/17), Python, Bash, SQL. \\
\textbf{Robotics:} ROS 2 Humble, Nav2, MoveIt, Gazebo, URDF/Xacro. \\
\textbf{AI/CV:} PyTorch, YOLOv8, OpenCV, PointCloud Library (PCL). \\
\textbf{Hardware:} NVIDIA Jetson, Raspberry Pi, LIDAR (Velodyne), Realsense.

% Section: Projects
\section{Projects}
\textbf{Autonomous Agricultural Mobile Manipulator (AgriBot)} \\
\textit{Capstone Project} \hfill \textit{Oct 2025 - Nov 2025}
\begin{itemize}
    \item Designed and simulated a mobile manipulator in Gazebo for strawberry harvesting.
    \item Implemented a ROS 2 Navigation stack with 95\% success rate in crop row following.
    \item Integration a Custom YOLOv8 detection node with MoveIt for visual servoing interactions.
    \item Optimized System Cycle time by 30\% using asynchronous behavior trees.
\end{itemize}
```

### 👨‍💻 Portfolio Logic (`portfolio/index.html`)

A static site using a simple grid layout.

```html
<!DOCTYPE html>
<html>
<head><title>Jane Doe | Robotics Engineer</title></head>
<body>
    <h1>Jane Doe</h1>
    <p>Robotics Engineer specializing in SLAM and Manipulation.</p>
    
    <h2>Featured Project: AgriBot</h2>
    <video controls src="agribot_demo.mp4" width="600"></video>
    <p>A full-stack ROS 2 solution for picking fruit.</p>
    <a href="https://github.com/janedoe/agribot">View Code on GitHub</a>
    
    <h2>Phase 4: ADAS Implementation</h2>
    <p>Lane Keeping and Obstacle Avoidance from scratch.</p>
</body>
</html>
```

---

## 🔬 Lab Exercise: "The Mock Interview"

### 1. Lab Objectives
- **Pair Up:** Find a partner (or use an AI chatbot).
- **Question 1:** "Explain how a Particle Filter works to a 5-year-old."
    *   *Ans:* "Imagine you are lost in a room. You throw 1000 confetti pieces on the map where you *might* be. As you move, you move the confetti. If you see a door, you throw away confetti that isn't near a door. Eventually, all confetti is in one pile. That's where you are."
- **Question 2:** "Write a C++ class for a PID controller."
    *   *Ans:* Implement `update(error, dt)`. Remember `integral += error * dt`. Handle `integral_windup`.

---

## 🚀 Project Steps

1.  **Clean GitHub:** Hide/Archive "Hello World" repos. Pin the "AgriBot" and "Phase 4 ADAS" repos.
2.  **Add License:** Ensure all code has an MIT License so employers know they can look at it safely.
3.  **Commit Graph:** Ensure your contribution graph has green squares (Evidence of consistency).

---

## 🐞 Debugging & Troubleshooting

### Common Hiring Failures

#### 1. "I know everything"
*   **Issue:** Listing "Expert in C++" when you've only used it for 1 year.
*   **Fix:** Be honest. "Proficient in C++". Expect to be grilled on Virtual Destructors if you say "Expert".

#### 2. "Generic Resume"
*   **Issue:** Sending the same resume to a Drone company and a Warehouse company.
*   **Fix:** **Tailor it.** For Drone: Highlight "Kalman Filters" and "3D Nav". For Warehouse: Highlight "A*" and "Multi-Agent".

---

## ⚡ Optimization: Keywords

ATS (Applicant Tracking Systems) filter resumes.
*   **Do:** Use standard spellings ("Object Oriented Programming", not "OOP").
*   **Do:** Mention specific tools ("Docker", "Git", "Linux").

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `const int* p` and `int* const p`? (Classic Interview Q).
    *   **A:** `const int*` = Pointer to a constant integer (Can't change value). `int* const` = Constant pointer to an integer (Can't change address).
2.  **Q:** Explain RAII.
    *   **A:** Resource Acquisition Is Initialization. Constructor acquires, Destructor releases (e.g., `std::lock_guard`). Prevents memory leaks.

### Challenge Task
> **Task:** "LeetCode Hard".
> 1. Solve "Median of Two Sorted Arrays".
> 2. Now optimize space complexity.
> 3. Now explain how this relates to merging sensor streams (Time Synchronization).

---

## 📚 Further Reading
- **Cracking the Coding Interview:** (The Bible).
- **Robotics-Worldwide:** Mailing list for jobs.

---

**Day 209 Complete**
