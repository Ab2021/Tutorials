# Day 91: Week 13 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 13: Humanoid Robotics

---

> **📝 Content Creator Instructions:**
> The most complex machine ever built.
> - **Goal:** Integrate Locomotion (Body) and VLA (Brain) and Teleop (Nervous System) into a unified demonstration.
> - **Code:** A `humanoid_bringup.launch.py` and a Capstone Video (simulated) script where the robot walks and waves.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Orchestrate** multiple controllers (Lower Body Walking + Upper Body Manipulation) safely.
2.  **Debug** Humanoid "Falls": Is it the ZMP Planner? Is it the State Estimator? Is it Physics instability?
3.  **Deploy** a "Socially Aware" Humanoid that interacts with people.

---

## 📚 Week 13 Review: The Humanoid Stack

| Day | Topic | Key Lesson | Tool |
|-----|-------|------------|------|
| **85** | **Platforms** | Floating Base URDFs & Simulation | `gazebo`, `urdf` |
| **86** | **Locomotion** | ZMP & Preview Control | `scipy`, `lipm` |
| **87** | **WBC** | Prioritized Task Hierarchy | `cvxpy`, `pinocchio` |
| **88** | **Safety** | Compliance & Collision Reflex | `ros2_control` |
| **89** | **Teleop** | Motion Retargeting | `mediapipe` |
| **90** | **Foundation Models** | VLA (Vision-Language-Action) | `LLaVA`, `RT-2` |

### The "Stack" Diagram
```mermaid
graph TD
    A[Human/LLM] --> B[Task Planner]
    B --> C[Behavior Tree]
    
    subgraph Upper Body
    C --> D[Arm Trajectory]
    D --> E[Inverse Kinematics]
    end
    
    subgraph Lower Body
    C --> F[Footstep Planner]
    F --> G[ZMP Preview Control]
    end
    
    E --> H[Whole Body Controller (QP)]
    G --> H
    H --> I[Joint Torques]
    I --> J[Sim/Hardware]
    J --> K[State Estimator]
    K --> H
```

---

## 🚀 Weekly Capstone: "The Service Droid"

**Scenario:** A humanoid waiter in a café.
**Mission:** "Greet customer, Walk to Table 4, Place Menu".
**Components:**
1.  **Locomotion:** Walk to `(x=5, y=0)` using ZMP.
2.  **HRI:** Detect Face. Wave Hand (Teleop playback or Joint Trajectory).
3.  **VLA:** "See table. Plan Place(Menu)".

### 🛠️ Project Structure
```text
week13_capstone/
├── launch/
│   └── full_system.launch.py
├── src/
│   ├── coordinator.py
│   └── walking_client.py
├── behavior_trees/
│   └── waiter.xml
└── config/
    └── wbc_params.yaml
```

### 👨‍💻 Coordinator Node (`src/coordinator.py`)

Manage theStateMachine.

```python
import rclpy
from rclpy.node import Node
# ... imports for Actions

class HumanoidCoordinator(Node):
    def __init__(self):
        super().__init__('coordinator')
        # Clients
        self.walk_client = ActionClient(self, WalkTo, 'walk_to')
        self.wave_client = ActionClient(self, PlayMotion, 'play_motion')
        
    def run_mission(self):
        # 1. Greet
        self.get_logger().info("Greeting...")
        self.wave_client.send_goal_async(PlayMotion.Goal(motion_name='wave_hello'))
        
        # 2. VLA Check (Mock)
        # instruction = "Go to table 4"
        # plan = vla.plan(image, instruction)
        
        # 3. Walk
        self.get_logger().info("Walking to Table 4...")
        goal = WalkTo.Goal()
        goal.target_x = 5.0
        goal.target_y = 0.0
        self.walk_client.send_goal_async(goal)
```

### 👨‍💻 Launch File Breakdown

```python
# full_system.launch.py
# 1. Robot State Publisher (URDF)
# 2. Gazebo (Physics)
# 3. Controller Manager (Load 'walking_controller' and 'arm_controller')
# 4. State Estimator (Floating Base Odometry)
# 5. Perception (Camera Node)
# 6. Mission Coordinator
```

---

## 📝 Self-Assessment Quiz

1.  **Locomotion:**
    *   What happens if the floor is slippery?
    *   **A:** The foot slips. The ZMP assumption (fixed contact) breaks. The robot falls. Requires "Slip Recovery" (Reflex).
2.  **Control:**
    *   Why use WBC instead of IK per limb?
    *   **A:** Because lifting a heavy arm shifts the CoM. An isolated Arm IK doesn't know this, but the WBC (managing CoM) will automatically lean the torso back to compensate.
3.  **AI:**
    *   How does VLA help?
    *   **A:** It allows "Open Vocabulary" interaction. You can say "Put it near the red cup" instead of hardcoding `(x,y)` coordinates.

---

## ⏭️ Look Ahead: Week 14
From the Body to the Brain (Hardware).
**Week 14: Edge AI Deployment.**
*   Running these massive models (LLaVA, Diffusion, WBC) on valid hardware.
*   NVIDIA Jetson Orin Deep Dive.
*   TensorRT Optimization.

---

**Week 13 Complete**
