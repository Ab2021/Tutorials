# Day 200: Planning & Control Implementation
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 29: Capstone Project Part 1

---

> **📝 Content Creator Instructions:**
> Don't run and gun. Stop, then shoot (pick).
> - **Focus:** Integrating Nav2 for mobility and MoveIt! for manipulation. Building the High-Level State Machine.
> - **Code:** `harvest_fsm.py`. A Python node using `rclpy` actions to sequence the mission (Drive $\to$ Detect $\to$ Pick $\to$ Basket).
> - **Concept:** Hierarchical Control (Deliberative vs Reactive).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** a Nav2 behavior tree for "Row Following" (Waypoint navigation).
2.  **Program** a MoveIt `MoveGroupInterface` python script to reach a target XYZ.
3.  **Implement** a Finite State Machine (FSM) to coordinate Navigation and Manipulation.
4.  **Handle** failure states (e.g., "Planning Failed" or "Fruit Dropped").

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-nav2-bringup ros-humble-moveit
pip install transitions # Python FSM library
```

### Prior Knowledge
- Nav2 Actions (Day 16).
- MoveIt API (Day 17).

---

## 📖 Theoretical Structure

### 🔹 The Harvest Cycle

1.  **IDLE:** Wait for start signal.
2.  **NAV_TO_REGION:** Drive to the start of the row.
3.  **SEARCH:** Drive slowly (0.1 m/s) scanning for fruits.
4.  **DETECTED:** Stop base.
5.  **APPROACH:** Arm moves to pre-grasp pose (offset 10cm).
6.  **GRASP:** Arm moves forward + Close Gripper.
7.  **RETRACT:** Pull back.
8.  **DEPOSIT:** Move to Basket + Open Gripper.

### 🔹 Two Brains problem

*   **Nav2:** Controls `cmd_vel`. Focus: "Avoid walls".
*   **MoveIt:** Controls `joint_trajectory`. Focus: "Avoid Self".
*   **Conflict:** If Arm moves while Base moves, the Center of Mass shifts, and TF tree is dynamic.
*   **Solution:** **Stop-and-Pick**. Easiest for V1.

---

## 💻 Implementation: The Harvest FSM

### 🛠️ Project Structure
```text
agribot_control/
├── src/
│   ├── harvest_fsm.py
│   └── arm_commander.py
└── launch/
    └── full_mission.launch.py
```

### 👨‍💻 Arm Commander (`src/arm_commander.py`)

Wrapper around MoveIt.

```python
import rclpy
from moveit_commander import MoveGroupCommander

class ArmClient:
    def __init__(self):
        self.group = MoveGroupCommander("ur5_arm")
        self.gripper = MoveGroupCommander("gripper")
        
    def go_to_pose(self, x, y, z):
        self.group.set_position_target([x, y, z])
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        return plan
    
    def open_gripper(self):
        # Assuming joint control for simplicity
        self.gripper.set_named_target("open")
        self.gripper.go(wait=True)
        
    def close_gripper(self):
        self.gripper.set_named_target("closed")
        self.gripper.go(wait=True)
```

### 👨‍💻 Harvest FSM (`src/harvest_fsm.py`)

Using `BasicNavigator` (Nav2) and `ArmClient`.

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from arm_commander import ArmClient
import time

class HarvestMission(Node):
    def __init__(self):
        super().__init__('harvest_mission')
        
        self.nav = BasicNavigator()
        self.arm = ArmClient()
        
        self.status = "SEARCHING"
        self.fruits_queue = []
        
        # Subscribe to Perception
        self.create_subscription(PoseArray, '/perception/fruits_3d', self.fruit_cb, 10)
        
        # Start Routine
        self.timer = self.create_timer(1.0, self.loop)
        
    def fruit_cb(self, msg):
        # Add new fruits to queue if not already there (simple dist check)
        for pose in msg.poses:
            # Check duplicates...
            self.fruits_queue.append(pose)
            
    def loop(self):
        if self.status == "SEARCHING":
            # Drive forward slowly
            # In Nav2, send waypoint X+1m
            # self.nav.goToPose(...)
            print("Searching...")
            
            if len(self.fruits_queue) > 0:
                self.nav.cancelTask() # STOP!
                self.status = "PICKING"
                
        elif self.status == "PICKING":
            target = self.fruits_queue.pop(0)
            print(f"Targeting Fruit at {target.position.x:.2f}, {target.position.y:.2f}")
            
            # 1. Pre-Grasp
            success = self.arm.go_to_pose(target.position.x - 0.1, target.position.y, target.position.z)
            if not success:
                print("Reach Unreachable")
                self.status = "SEARCHING"
                return

            self.arm.open_gripper()
            
            # 2. Grasp
            self.arm.go_to_pose(target.position.x, target.position.y, target.position.z)
            self.arm.close_gripper()
            
            # 3. Retract
            self.arm.go_to_pose(0.3, 0.0, 0.5) # Home/Retract
            
            # 4. Basket
            self.arm.go_to_pose(0.0, -0.4, 0.3) # Basket location relative to base
            self.arm.open_gripper()
            
            if len(self.fruits_queue) == 0:
                self.status = "SEARCHING"

def main():
    rclpy.init()
    node = HarvestMission()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The First Pick"

### 1. Lab Objectives
- **Setup:** Place a "Fruit" model at $(x=1.0, y=0.0, z=0.5)$ in Gazebo.
- **Run:** `ros2 launch agribot_control full_mission.launch.py`.
- **Observe:**
    1.  Robot drives towards it.
    2.  Detector publishes topic.
    3.  Robot stops.
    4.  Arm reaches out.
    5.  Gripper closes.
    6.  Arm dumps it in the carry-bin (on the back of the Husky).

---

## 🚀 Project Steps

1.  **Nav2 Config:** Create `nav2_params.yaml`. Ensure `robot_radius` accounts for the arm sticking out!
2.  **MoveIt Config:** Run `moveit_setup_assistant`. Define "arm" group and "gripper" group. Disable collisions between adjacent links.
3.  **Basket:** Define a collision object "Basket" attached to the robot base so MoveIt doesn't think the basket is an obstacle (or does, but allows putting things *in* it).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Planning Timeouts"
*   **Cause:** IK solver (KDL) stuck.
*   **Fix:** Switch to TRAC-IK (much faster). Increase planning time allowed to 5.0s.

#### 2. "Camera Occlusion"
*   **Scene:** Arm moves to pick fruit, but blocks the camera view. Perception loses the fruit.
*   **Fix:** **Memory.** The perception node should remember the location. Once the sequence "PICKING" starts, ignore camera updates (Blind Grasp).

---

## ⚡ Optimization: Look-Ahead

Instead of Stop-Pick-Drive-Stop-Pick.
*   **Pipeline:** Scan the whole row first. Build a map of all fruits. Then optimize the picking order (TSP). Then drive back and pick efficiently.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why `nav2_simple_commander`?
    *   **A:** It's a Python wrapper API that makes sending Nav2 goals strictly easier than raw Action Clients.
2.  **Q:** Difference between `go_to_pose` (Cartesian) and `go_to_joint_state`?
    *   **A:** Pose uses IK to find joint angles. Joint state is direct. Use Pose for picking, Joint for "Home".

### Challenge Task
> **Task:** "Visual Servoing".
> 1. Don't trust the initial position.
> 2. As the arm executes the approach, keep reading the camera.
> 3. Adjust the `set_position_target` in real-time ($30Hz$) to correct for robot base drift.

---

## 📚 Further Reading
- **MoveIt Tutorials:** "Python Interface".
- **Nav2:** "Writing a First Navigator".

---

**Day 200 Complete**
