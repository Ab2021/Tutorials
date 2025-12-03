# Day 162: Project Planning & Architecture
## Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking

---

> **📝 Day 162 Focus:**
> Welcome to the **Final Capstone**. Your mission: Build an **Autonomous Valet Parking (AVP)** system. The user drops the car at the entrance, and the car finds a spot and parks itself. Today, we plan the architecture.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the Operational Design Domain (ODD) for AVP.
2.  **Draft** System Requirements (Functional & Safety).
3.  **Design** the Software Architecture (ROS 2 Nodes & Topics).
4.  **Specify** Sensor Configuration (Cameras, Ultrasonics).
5.  **Create** an Interface Control Document (ICD).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **All Previous Weeks:** Perception, Localization, Planning, Control.

### Hardware Requirements
-   **None:** Planning phase.

### Software Stack
-   **Tools:** Draw.io / Mermaid (Diagrams).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The ODD (Operational Design Domain)

-   **Environment:** Parking Garage (GPS denied, Low speed < 10 km/h).
-   **Lighting:** Artificial light, Shadows.
-   **Traffic:** Pedestrians, other cars backing out.
-   **Maneuvers:** 90-degree turn, Reverse parking.

### 🔹 Part 2: System Requirements

1.  **REQ-AVP-001:** The system shall detect empty parking spots with > 95% accuracy.
2.  **REQ-AVP-002:** The system shall localize within 10cm accuracy without GPS.
3.  **REQ-AVP-003:** The system shall stop within 0.5m of a pedestrian.
4.  **REQ-AVP-004:** The system shall park within the lines (lateral error < 5cm).

### 🔹 Part 3: Sensor Suite

-   **4x Fisheye Cameras:** Surround View (360°).
-   **12x Ultrasonics:** Close-range obstacle detection (0-3m).
-   **IMU/Wheel Odometry:** Dead Reckoning.
-   **No Lidar:** (Cost constraint for L2/L3 parking).

---

## 💻 Implementation: Architecture Design

**Scenario:**
-   Design the ROS 2 Graph for the AVP system.

### 🛠️ Setup
Create `week24_capstone` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week24_capstone
mkdir -p week24_capstone/docs
```

### 👨‍💻 Code: Architecture Diagram (Mermaid)

Create `docs/architecture.md`.

```mermaid
graph TD
    subgraph Sensors
        Cam[Camera Driver]
        USS[Ultrasonic Driver]
        Odom[Odometry Driver]
    end

    subgraph Perception
        Seg[Semantic Segmentation<br/>(Lines/Free Space)]
        Obj[Object Detection<br/>(Cars/Peds)]
        Slot[Parking Slot Detector]
        Fusion[Occupancy Grid Map]
    end

    subgraph Localization
        EKF[EKF Fusion<br/>(Odom + Visual)]
    end

    subgraph Planning
        Global[Global Planner<br/>(A* on Grid)]
        Park[Parking Planner<br/>(Hybrid A* / Reeds-Shepp)]
        FSM[Behavior State Machine]
    end

    subgraph Control
        MPC[MPC Controller]
    end

    Cam --> Seg
    Cam --> Obj
    Cam --> Slot
    USS --> Fusion
    Seg --> Fusion
    
    Odom --> EKF
    
    Fusion --> Global
    Fusion --> Park
    EKF --> Global
    EKF --> Park
    EKF --> MPC
    
    Global --> FSM
    FSM --> Park
    Park --> MPC
    MPC --> Actuators
```

### 👨‍💻 Code: Interface Control Document (ICD)

Create `docs/interfaces.yaml`.

```yaml
topics:
  /perception/occupancy_grid:
    type: nav_msgs/OccupancyGrid
    freq: 10 Hz
    desc: Local map of obstacles (static + dynamic).

  /perception/parking_spots:
    type: vision_msgs/Detection2DArray
    freq: 10 Hz
    desc: Bounding boxes of empty spots.

  /localization/pose:
    type: geometry_msgs/PoseStamped
    freq: 50 Hz
    desc: Vehicle position in Map frame.

  /planning/trajectory:
    type: nav_msgs/Path
    freq: 10 Hz
    desc: Planned path to the parking spot.

  /control/cmd_vel:
    type: geometry_msgs/Twist
    freq: 50 Hz
    desc: Steering and Throttle commands.
    
  /system/state:
    type: std_msgs/String
    freq: 1 Hz
    desc: FSM State (IDLE, SEARCHING, PARKING, PARKED).
```

---

## 🔬 Lab Exercise: Project Setup

### Lab Objectives
1.  **Create the Package:** `week24_capstone`.
2.  **Create Placeholders:**
    -   `perception_node.py`
    -   `planning_node.py`
    -   `control_node.py`
3.  **Create Launch File:** `avp.launch.py` that starts all empty nodes.
4.  **Verify Graph:** Run `rqt_graph` and ensure it matches your Mermaid diagram.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Scope Creep
**Symptom:** "Let's also handle Valet Retrieval via Phone App!"
**Cause:** Ambition.
**Solution:** Stick to the MVP (Minimum Viable Product). Drop-off $\to$ Park. Retrieval is Phase 2.

#### 2. Interface Mismatch
**Symptom:** Planner expects `PoseWithCovariance`, Localization sends `PoseStamped`.
**Cause:** Poor ICD.
**Solution:** Define `.msg` files early and stick to them.

---

## ⚡ Optimization & Best Practices

### 1. Coordinate Frames (REP-105)
-   `map`: Fixed frame (Parking Lot entrance).
-   `odom`: Drifting frame (Wheel encoders).
-   `base_link`: Vehicle center (Rear axle).
-   **Crucial:** Parking requires high precision. Ensure TF tree is perfect.

### 2. Simulation First
-   Don't build the car yet.
-   Build the **Gazebo/CARLA** world first.
-   If you can't park in Sim, you can't park in Real Life.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is GPS denied in AVP?
    *   **A:** Parking garages are concrete bunkers. GPS signal is blocked or multipath is terrible.
2.  **Q:** What is the hardest part of AVP?
    *   **A:** Localization (without GPS) and precise Maneuvering (Reverse parking with 10cm clearance).
3.  **Q:** What is "Hybrid A*"?
    *   **A:** A path planning algorithm that considers the vehicle's kinematics (turning radius) while searching the grid. Essential for parking.

### Challenge Task
**Task:** Map Analysis.
1.  Download a floor plan of a parking garage.
2.  Measure the aisle width and spot size.
3.  Calculate if your vehicle (Tesla Model 3 dimensions) can physically make the turn into the spot. (Ackermann geometry).

---

## 📚 Further Reading & References
-   [Hybrid A* Algorithm](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf)
-   [ROS 2 REP-105 (Coordinate Frames)](https://www.ros.org/reps/rep-0105.html)

---

**Day 162 Complete** | Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking
