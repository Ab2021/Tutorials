# Day 197: Project Planning & Requirements
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 29: Capstone Project Part 1

---

> **📝 Content Creator Instructions:**
> The Final Boss.
> - **Focus:** Designing the "Agri-Bot 5000". Autonomous Harvesting Robot.
> - **Code:** `system_architecture.yaml`, `requirements.md`. Defining nodes, topics, and hardware specs.
> - **Concept:** Systems Engineering (V-Model).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** precise requirements for a complex mobile manipulator.
2.  **Architect** the ROS 2 Network (Nodes, Topics, Actions).
3.  **Select** appropriate hardware (LIDAR vs Depth Cam, Differential vs Ackerman).
4.  **Create** a Gantt chart for the 2-week implementation sprint.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Paper & Pencil.
- (Virtual) Gazebo Environment setup.

### Software Environment
```bash
# Install diagram tools if desired
pip install diagrams
```

### Prior Knowledge
- All Phases 1-5.

---

## 📖 Theoretical Mission Brief

### 🔹 The Problem: "The Strawberry Crisis"
Labor shortage is hitting farms. We need a robot that can:
1.  **Navigate** rows of crops (GPS/Lidar).
2.  **Detect** ripe strawberries (CV).
3.  **Manipulate** a gripper to pick them without bruising (Soft Control).
4.  **Deposit** them in a basket.

### 🔹 Requirements Traceability
*   **REQ-001:** Robot shall traverse 100m row with deviation < 10cm. (Nav2).
*   **REQ-002:** Robot shall identify ripe fruit > 80% confidence. (YOLOv8).
*   **REQ-003:** Picking cycle time < 5s per fruit. (MoveIt).
*   **REQ-004:** Battery life > 4 hours. (Energy Budget).

---

## 💻 Implementation: System Architecture

We define the blueprint before writing C++.

### 🛠️ Project Structure
```text
start_capstone/
├── design/
│   ├── graph.png (ROS Graph)
│   └── architecture.yaml
├── docs/
│   └── requirements.md
└── launch/
    └── system_bringup.launch.py (Placeholder)
```

### 👨‍💻 Architecture Definition (`design/architecture.yaml`)

```yaml
system_name: "AgriBot_V1"

nodes:
  - name: "/perception/fruit_detector"
    package: "agribot_vision"
    type: "yolo_node"
    input_topics: ["/camera/rgb/image_raw"]
    output_topics: ["/perception/fruits_detected"]

  - name: "/navigation/nav_stack"
    package: "nav2_bringup"
    input_topics: ["/scan", "/odom", "/map"]
    output_topics: ["/cmd_vel"]

  - name: "/manipulation/arm_commander"
    package: "moveit_cpp"
    input_topics: ["/perception/fruits_detected"]
    output_topics: ["/joint_trajectory"]

  - name: "/hardware/base_controller"
    package: "ros2_control"
    input_topics: ["/cmd_vel", "/joint_trajectory"]
    output_topics: ["/odom", "/joint_states"]

hardware:
  lidar: "Velodyne VLP-16 (Simulated)"
  camera: "Intel Realsense D435"
  arm: "UR5e (6 DoF)"
  base: "Husky (4 Wheel Diff Drive)"
```

### 👨‍💻 Python Diagram Generator

Generating the ROS Graph programmatically.

```python
from diagrams import Diagram, Cluster
from diagrams.programming.flowchart import Action, Inspection, Database
from diagrams.custom import Custom

# Pseudo-code to visually render the YAML above
print("Generating Architecture Diagram...")
# (Requires Graphviz, usually skipped in pure text tutorials, 
# but conceptualizing the flow is key)
```

### 👨‍💻 Requirements Document (`docs/requirements.md`)

```markdown
# AgriBot Requirements

## 1. Perception
*   **P.1:** Must distinguish Red (Ripe) vs Green (Unripe) fruit.
*   **P.2:** Must estimate 3D pose (x,y,z) of fruit with < 5mm error for gripper.

## 2. Navigation
*   **N.1:** Crop Row Following. Can rely on Lidar detecting plants on Left/Right.
*   **N.2:** Obstacle Stop. Humans may walk in row.

## 3. Manipulation
*   **M.1:** Visual Servoing. Update target as arm moves.
*   **M.2:** Verify Pick. Force sensor detecting weight change.

## 4. Safety
*   **S.1:** E-Stop must be hardware based.
*   **S.2:** Max velocity 1.0 m/s.
```

---

## 🔬 Lab Exercise: "The Mockup"

### 1. Lab Objectives
- **Setup:** Create the `agribot_ws` workspace.
- **Import:** Import the URDFs for Husky and UR5.
- **Tf:** Configure the Static Transform Publisher (`base_link` -> `arm_base_link`).
- **Visualize:** Open RViz2 and verify the robot looks like a robot, not a pile of parts.

---

## 🚀 Project Plan (Gantt)

*   **Day 197:** Planning (Done).
*   **Day 198:** Simulation World (Gazebo Farm).
*   **Day 199:** Perception Pipeline (Fruit Detect).
*   **Day 200:** Navigation (Row Following).
*   **Day 201:** Manipulation (Pick & Place).
*   **Day 202:** Integration (State Machine).
*   **Day 203:** Optimization & Testing.

---

## 🐞 Debugging & Troubleshooting

### Common Design Flaws

#### 1. "Scope Creep"
*   **Issue:** "Let's also make it prune leaves and spray pesticides!"
*   **Fix:** **No.** Stick to the MVP (Minimum Viable Product). One task: Pick Red Balls.

#### 2. "Wrong Sensors"
*   **Issue:** Using 2D Lidar for fruit detection.
*   **Fix:** Fruits are small and inside foliage. You need 3D Depth Cameras (RGB-D) close up. Lidar is for the Base, not the Hand.

---

## ⚡ Optimization: Compute Budget

*   YOLOv8: 50ms (GPU).
*   Nav2: 20ms (CPU).
*   MoveIt: 100ms (CPU burst).
*   **Total:** Can runs on a Jetson Orin Nano? Yes.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "URDF"?
    *   **A:** Universal Robot Description Format. The XML file describing the robot's kinematics and geometry.
2.  **Q:** Why separate Perception, Nav, and Manip into nodes?
    *   **A:** Modular architecture. If vision crashes, the robot shouldn't drive off a cliff (Nav keeps running).

### Challenge Task
> **Task:** "Cost Analysis".
> 1. Estimate the BoM (Bill of Materials).
> 2. Husky ($20k) + UR5 ($30k) + Sensors ($5k) = $55k.
> 3. Too expensive for farmers? Can you design a version for $5k? (Custom chassis, Kinova arm?).

---

## 📚 Further Reading
- **ROS 2 Design Patterns:** Official architecture guides.
- **FarmBot:** Open source CNC farming.

---

**Day 197 Complete**
