# Day 168: Week 24 Review & Project (Final Demo)
## Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking

---

> **📝 Day 168 Focus:**
> You did it. You built an Autonomous Valet Parking system. Today is **Demo Day**. We polish the system, record a video, analyze the metrics, and document the journey. This is your portfolio piece.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Execute** a flawless AVP demo in CARLA.
2.  **Record** a high-quality video with Rviz overlays.
3.  **Analyze** performance metrics (Parking Accuracy, Time to Park).
4.  **Document** the system architecture and lessons learned.
5.  **Reflect** on the gap between this prototype and a production L4 system.

---

## 📚 Week 24 Review

### 1. The Architecture
-   **Map:** Semantic Map with Slots and Aisles.
-   **Perception:** 360° Vision + Ultrasonics.
-   **Planning:** Hybrid A* for non-holonomic paths.
-   **Control:** MPC for precise low-speed maneuvering.
-   **Mission:** State Machine orchestration.

### 2. The Result
-   A system that can navigate a parking lot, find a spot, and reverse park without human intervention.

---

## 🛠️ Capstone Project: The Final Report

**Goal:** Create a `README.md` and a Video for your GitHub portfolio.

### 📊 Metric Analysis

Run the parking scenario 10 times and record:

| Run | Success? | Lateral Error (cm) | Heading Error (deg) | Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Yes | 3.2 | 0.5 | 45 |
| 2 | Yes | 4.1 | 0.8 | 48 |
| 3 | No (Hit Curb) | - | - | - |
| ... | ... | ... | ... | ... |

**Analysis:**
-   **Success Rate:** 90%.
-   **Mean Accuracy:** 3.8 cm.
-   **Bottleneck:** Perception latency caused the curb hit in Run 3.

### 🎥 Video Production

1.  **Screen Recorder:** OBS Studio.
2.  **Layout:**
    -   **Left:** CARLA Spectator View (The "Drone" shot).
    -   **Right:** Rviz (The "Brain" view - Lidar, Path, Costmap).
    -   **Bottom:** Terminal output (State transitions).
3.  **Script:**
    -   "System Idle. User requests parking."
    -   "Mapping phase. Localizing..."
    -   "Slot Detected. Planning path..."
    -   "Reversing. Note the steering corrections."
    -   "Parked. Error is 3cm."

### 📝 Documentation (`README.md`)

```markdown
# Autonomous Valet Parking (AVP) System

## Overview
A ROS 2 based Level 4 AVP system simulated in CARLA. Capable of searching for parking spots and performing reverse parking maneuvers using Hybrid A* and MPC.

## Architecture
- **Perception:** 4x Fisheye Cameras (IPM), 12x Ultrasonics.
- **Localization:** EKF Fusion (Odom + IMU + Map Matching).
- **Planning:** Hybrid A* with Reeds-Shepp heuristics.
- **Control:** Model Predictive Control (MPC) with gear management.

## Performance
- **Success Rate:** 90% (n=10).
- **Parking Accuracy:** < 5cm.
- **Max Speed:** 10 km/h.

## How to Run
1. `./CarlaUE4.sh`
2. `ros2 launch week24_capstone avp.launch.py`
```

---

## 🧠 Comprehensive Assessment (The Final Exam)

### Section 1: System Design
1.  **Q:** Why did we choose MPC over PID for parking?
    *   **A:** PID cannot handle the multi-variable constraints (Steering + Speed) and the non-holonomic kinematics (Reversing) effectively. MPC predicts the future trajectory and optimizes for it.
2.  **Q:** How does the system handle a moving pedestrian?
    *   **A:** The Perception layer puts them in the Costmap. The Planner (or Safety Monitor) sees the occupied cells and stops the car.

### Section 2: Future Work
3.  **Q:** What is missing for a real product?
    *   **A:**
        -   **Safety:** ISO 26262 ASIL-D certification.
        -   **Redundancy:** Backup Compute, Backup Power.
        -   **V2I:** Communication with the Garage infrastructure (Booking spots).
        -   **SLAM:** Real-time mapping updates (Construction in garage).

---

## 🏆 Conclusion

Congratulations! You have completed Phase 4.
You have gone from basic ROS 2 nodes to a full-blown Autonomous Driving stack.
You understand Perception, Localization, Planning, Control, and the rigorous Testing required to make it safe.

**Next Steps:**
-   **Phase 5:** Advanced Topics (V2X, Fleet Management, Remote Teleoperation).
-   **Career:** Polish this project. Put it on GitHub. Show it to recruiters. This is real engineering.

---

**Day 168 Complete** | Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking
