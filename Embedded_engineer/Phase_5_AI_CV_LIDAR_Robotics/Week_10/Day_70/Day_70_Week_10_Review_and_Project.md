# Day 70: Week 10 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 10: Robot Manipulation & Grasping

---

> **📝 Content Creator Instructions:**
> We moved from "Move to X" to "Understand, Grasp, and Feel X".
> - **Goal:** Synthesize everything: MTC for planning, VLM for semantics, GraspNet for pose, and Impedance for safety.
> - **Code:** A `main_executive.py` that orchestrates the entire "Lab Assistant" mission.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architect** a complete manipulation pipeline integrating Perception, Planning, and Control.
2.  **Orchestrate** ROS 2 Actions: `Navigate` -> `Detect` -> `Plan` -> `Execute`.
3.  **Troubleshoot** common integration failures (TF Latency, Controller Conflicts).

---

## 📚 Week 10 Review: The Manipulation Stack

| Day | Topic | Key Lesson | Tool |
|-----|-------|------------|------|
| **64** | **MoveIt 2 Deep Dive** | Architecture & Real-Time Servo | `moveit_cpp`, `servo` |
| **65** | **MTC** | Task Logic (Pick = Approach + Grasp + Retreat) | `Task Constructor` |
| **66** | **Neural Grasping** | Sampling grasps for unknown objects | `GraspNet`, `PointNet` |
| **67** | **Dexterity** | Multi-finger Synergies & Grasp Matrices | `BiDexHand`, `Allegro` |
| **68** | **VLM Grasping** | "Pick the red one" (Semantics) | `OWL-ViT`, `CLIP` |
| **69** | **Force Control** | Soft interaction (Impedance) | `ros2_control` |

### The "Stack" Diagram
```mermaid
graph TD
    A[User Command: 'Pick Red Cup'] --> B[VLM (OWL-ViT)]
    B --> C[ROI: 2D Box]
    C --> D[GraspNet (3D Logic)]
    D --> E[MTC (Task Planning)]
    E --> F[Hybrid Planner (OMPL)]
    F --> G[Impedance Controller]
    G --> H[Robot Hardware]
    H -- Force Feedback --> G
```

---

## 🚀 Weekly Capstone: "The Intelligent Lab Assistant"

**Scenario:** Ideally a Mobile Manipulator (Tiago/Fetch), but a fixed arm on a table works too.
**Task:**
1.  **Setup:** A table with a Test Tube (Fragile), a Hammer (Heavy), and a Sponge (Soft).
2.  **Instruction:** "Clean the table" (Pick Sponge) OR "Smash the rock" (Pick Hammer).
3.  **Execution:**
    *   **Perception:** VLM identifies the object.
    *   **Grasp:** Neural Grasp Sampler finds a grasp.
    *   **Planning:** MTC plans the pick.
    *   **Control:**
        *   If Picking Hammer: Switch to **Position Control** (Stiff).
        *   If Picking Sponge: Switch to **Impedance Control** (Soft). The prompt implies property? Or VLM infers material?
        *   *Simplification:* Hardcode material mapping based on class name.

### 🛠️ Project Structure
```text
week10_capstone/
├── config/
│   └── objects.yaml (Class -> Stiffness Mapping)
├── src/
│   ├── capability_server.py
│   └── mission_executive.py
└── launch/
    └── system.launch.py
```

### 👨‍💻 Mission Executive (`src/mission_executive.py`)

Using `py_trees` (Behavior Trees) or a simple State Machine to sequence the actions.

```python
import rclpy
from rclpy.node import Node
# ... Imports for Actions

class LabAssistant(Node):
    def __init__(self):
        super().__init__('lab_assistant')
        
        # Action Clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.pick_client = ActionClient(self, PickObject, 'pick_object') # MTC Wrapper
        self.vlm_client = ActionClient(self, DetectObject, 'detect_object')
        
    def execute_command(self, prompt):
        # 1. Detect
        target_name, roi = self.call_vlm(prompt)
        
        # 2. Lookup Material Property
        stiffness = self.get_stiffness(target_name)
        
        # 3. Configure Controller
        self.set_controller_mode(stiffness)
        
        # 4. Pick
        self.call_mtc_pick(target_name, roi)
        
    def get_stiffness(self, name):
        if "sponge" in name: return "impedance_soft"
        if "hammer" in name: return "position_stiff"
        if "tube" in name: return "impedance_precise"
        return "position_default"

    def set_controller_mode(self, mode):
        # Service call to controller_manager to switch controllers
        pass
```

### 👨‍💻 Controller Switching Logic

This is the critical "System Integration" part often missed.
*   **Safety:** You cannot switch controllers while moving.
*   **Sequence:**
    1.  Stop Robot (Vel = 0).
    2.  `switch_controllers(stop=['joint_traj'], start=['impedance'])`.
    3.  Sleep 0.1s.
    4.  Send new Command.

---

## 📝 Self-Assessment Quiz

1.  **Architecture:**
    *   Why use MTC instead of a simple Python script?
    *   **A:** Error Handling. If the grasp is valid but the retreat is blocked, MTC knows *before* you move. A script would move, grasp, and then get stuck.
2.  **Control:**
    *   Why is Impedance Control crucial for "Wiping a table"?
    *   **A:** Position control would either not touch the table (detached) or press too hard (break robot). Impedance maintains contact force $F_z$.
3.  **Perception:**
    *   Effect of Lighting on VLM?
    *   **A:** High sensitivity. CLIP works well, but depth sensors (Time-of-Flight) fail on black/shiny surfaces.

---

## ⏭️ Look Ahead: Week 11
We leave the Tabletop and go Mobile.
**Week 11: Navigation 2 (Nav2) Mastery.**
*   We know how to manipulate. Now we need to move between tables.
*   Nav2 Stack, Behavior Trees (BT.CPP), Costmaps, and Outdoor GPS.
*   We will build a "Campus Delivery Robot".

---

**Week 10 Complete**
