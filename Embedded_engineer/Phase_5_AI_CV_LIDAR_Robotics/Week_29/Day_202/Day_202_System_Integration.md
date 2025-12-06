# Day 202: System Integration
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 29: Capstone Project Part 1

---

> **📝 Content Creator Instructions:**
> The moment of truth.
> - **Focus:** Integrating all subsystems (Sim, Perception, Nav, Control, Learning) into a single Launch file. System Verification.
> - **Code:** `agribot_bringup` package. `sanity_check.sh`.
> - **Concept:** Integration Testing.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Create** a master `bringup` launch file that includes all sub-launches.
2.  **Verify** the TF tree is connected and healthy (`tf2_tools`).
3.  **Run** a full autonomous picking mission in Gazebo.
4.  **Debug** race conditions (e.g., node starting before params are ready).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-rqt-tf-tree
```

### Prior Knowledge
- Launch Files (Day 9).

---

## 📖 Theoretical Verification

### 🔹 The "Swiss Cheese" Model of Failure

Each layer has holes (flaws).
1.  **Vision:** Misses fruit 10% of time.
2.  **Nav:** Misses goal by 5cm 5% of time.
3.  **Control:** Gripper slips 5% of time.
4.  **System:** Success rate = $0.9 \times 0.95 \times 0.95 = 0.81$.
5.  **Result:** 1 in 5 fruits fail. We need to handle this failure gracefully (Retry logic).

### 🔹 Start-up Order

1.  **Hardware/Sim:** Gazebo, Robot State Publisher. (Must exist first).
2.  **Nav2/MoveIt:** Planning Stacks. (Need TF/Descriptions).
3.  **Perception:** Sensors. (Need topic flow).
4.  **Application:** The FSM (`harvest_mission`). (Needs everything else).

---

## 💻 Implementation: The Master Switch

### 🛠️ Project Structure
```text
agribot_bringup/
├── launch/
│   ├── sim_complete.launch.py
│   └── hardware_complete.launch.py
└── scripts/
    └── sanity_check.py
```

### 👨‍💻 Complete Launch (`launch/sim_complete.launch.py`)

Using `IncludeLaunchDescription` to compose the stack.

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_sim = get_package_share_directory('agribot_description')
    pkg_nav = get_package_share_directory('agribot_navigation') # Hypothetical
    pkg_perc = get_package_share_directory('agribot_perception')
    pkg_ctrl = get_package_share_directory('agribot_control')
    
    # 1. Sim (Gazebo + Robot State Pub + Controllers)
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_sim, 'launch', 'gazebo.launch.py'))
    )
    
    # 2. Nav2 (Wait 10s for Sim)
    nav_launch = TimerAction(
        period=10.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_nav, 'launch', 'navigation.launch.py'))
        )]
    )
    
    # 3. Perception (Wait 5s)
    perc_launch = TimerAction(
        period=5.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_perc, 'launch', 'perception.launch.py'))
        )]
    )
    
    # 4. Learning Service
    learn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('agribot_learning'), 'launch', 'learning.launch.py'))
    )

    # 5. Mission FSM (Wait 20s for everything)
    mission_launch = TimerAction(
        period=20.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_ctrl, 'launch', 'full_mission.launch.py'))
        )]
    )
    
    return LaunchDescription([
        sim_launch,
        nav_launch,
        perc_launch,
        learn_launch,
        mission_launch
    ])
```

### 👨‍💻 Sanity Check (`scripts/sanity_check.py`)

A script to run *before* the mission to ensure green lights.

```python
import rclpy
from rclpy.node import Node
import time

class SystemCheck(Node):
    def __init__(self):
        super().__init__('sanity_checker')
        self.node_names = [
            '/fruit_detector', 
            '/move_group', 
            '/bt_navigator',
            '/grasp_service'
        ]
        
    def check_nodes(self):
        print("Checking System Health...")
        current_nodes = self.get_node_names()
        all_good = True
        
        for n in self.node_names:
            if n[1:] in current_nodes: # Remove leading slash logic
                print(f"[OK] {n}")
            else:
                print(f"[FAIL] {n} missing!")
                all_good = False
                
        # Check TF (Quick hack check)
        # buffer.lookup_transform('map', 'ee_link', ...)
        
        return all_good

def main():
    rclpy.init()
    chk = SystemCheck()
    if chk.check_nodes():
        print("SYSTEM GREEN. GO FOR MISSION.")
    else:
        print("SYSTEM RED. ABORT.")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Full Run"

### 1. Lab Objectives
- **Launch:** `ros2 launch agribot_bringup sim_complete.launch.py`.
- **Monitor:** Open `rqt_graph`. Is perception connected to FSM? Is FSM connected to Nav?
- **Watch:** The robot wakes up. Unfolds arm. Drives 5m. Detects red sphere. Picks it up. Drops in bucket.
- **Fail:** If it fails, where? (Nav suck? Grasp miss? TF error?).

---

## 🚀 Project Steps

1.  **Logging:** Ensure all nodes log to `/rosout`. Use `rqt_console` to filter ERRORs.
2.  **Parameters:** Use a central `params.yaml` loaded by the bringup launch to set global vars (e.g., `use_sim_time: true`).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Process Died" (Exit Code -9)
*   **Cause:** OOM (Out of Memory). Running Gazebo + Rviz + MoveIt + YOLO on a laptop with 8GB RAM.
*   **Fix:** Close Web Browser. Run Gazebo headless (`gui:=false`).

#### 2. "Transform Timeout"
*   **Cause:** `use_sim_time` mismatch. Some nodes using Wall Time, some Sim Time.
*   **Fix:** Ensure **EVERY** node in **EVERY** launch file has `parameters=[{'use_sim_time': True}]`.

---

## ⚡ Optimization: Behavior Trees

The Python FSM is getting messy (`if status == ...`).
*   **Upgrade:** Replace `harvest_fsm.py` with `Groot` (Behavior Tree).
*   **Structure:**
    *   Sequence:
        *   NavToRow
        *   Fallback (Retry Loop):
            *   Sequence (Pick):
                *   Detect
                *   Grasp
            *   Recovery (Back up 10cm)

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why `TimerAction` in launch?
    *   **A:** To stagger startup. Prevents CPU spike from freezing the OS, and ensures servers (Nav/MoveIt) are ready before clients (FSM) try to connect.
2.  **Q:** What is `sanity_check` good for?
    *   **A:** Pre-flight checklist. Saves time debugging a mission that was doomed to fail because the camera node crashed on boot.

### Challenge Task
> **Task:** "Continuous Integration".
> 1. Write a GitHub Action.
> 2. On Push: Build workspace.
> 3. Run `sanity_check.py` against a text fixture.

---

## 📚 Further Reading
- **ROS 2 Launch:** "Using Event Handlers".
- **BehaviorTree.CPP:** "The standard for complex robot logic".

---

**Day 202 Complete**
