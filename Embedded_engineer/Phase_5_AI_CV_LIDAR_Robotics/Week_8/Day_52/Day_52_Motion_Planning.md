# Day 52: Motion Planning (MoveIt 2)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 8: Advanced Manipulation

---

> **📝 Content Creator Instructions:**
> We have IK (Pose $\to$ Angles). But how do we get there without hitting the table?
> - **Focus:** MoveIt 2 Framework, Planning Scene Monitor, OMPL (RRTConnect), and Trajectory Execution.
> - **Code:** Setting up a Franka Emika Panda arm and scripting a "Pick" motion.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** a robot for MoveIt using the Setup Assistant (SRDF generation).
2.  **Interface** with the `MoveGroupInterface` in ROS 2 (C++/Python).
3.  **Add** Collision Objects (e.g., a Table) to the Planning Scene.
4.  **Execute** a collision-free motion plan using OMPL planners.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulated).

### Software Environment
```bash
sudo apt install ros-humble-moveit
sudo apt install ros-humble-panda-moveit-config
```

### Prior Knowledge
- ROS 2 Actions.
- RRT* (Day 15).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The MoveIt Architecture

MoveIt is a monolith that wraps:
1.  **Planning Scene Monitor:** Keeps track of "World" (Octomap/Boxes) and "Robot State" (Joints).
2.  **Planning Pipeline (OMPL):** Samples valid paths.
3.  **Trajectory Processing:** Time Parameterization (velocity/acceleration scaling).
4.  **Controller Manager:** Speaks to hardware drivers (`FollowJointTrajectory` action).

### 🔹 Part 2: Configuration (SRDF)

The Semantic Robot Description Format (SRDF) complements URDF.
*   **Groups:** Defines chains (e.g., `panda_arm`, `hand`).
*   **Virtual Joints:** Does the robot fly or is it bolted to `world`?
*   **Passive Joints:** Casters that don't need planning.
*   **Self-Collisions:** Disable collision checking for adjacent links (faster).

### 🔹 Part 3: Planners (OMPL)

Open Motion Planning Library.
*   **RRTConnect:** Standard. Bi-directional RRT. Very fast finding *a* path. Not optimal.
*   **RRT*:** Optimal path (shortest), but slower.
*   **BiTRRT:** Bi-directional T-RRT. Good for costmaps.

---

## 💻 Implementation: MoveIt Commander

Scripting the robot in Python.

### 🛠️ Project Structure
```text
day52_moveit/
├── launch/
│   └── demo.launch.py
├── scripts/
│   ├── simple_move.py
│   └── box_pick.py
└── CMakeLists.txt
```

### 👨‍💻 Launch File (`launch/demo.launch.py`)

Using the pre-made Panda config.

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    moveit_config = get_package_share_directory('panda_moveit_config')
    
    # Launch MoveIt Demo (RvIz + Fake Controller)
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit_config, 'launch', 'demo.launch.py')
            )
        )
    ])
```

### 👨‍💻 Python Interface (`scripts/simple_move.py`)

```python
import rclpy
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from geometry_msgs.msg import Pose

def main():
    rclpy.init()
    
    # 1. Initialize Interfaces
    robot = RobotCommander()
    scene = PlanningSceneInterface()
    group = MoveGroupCommander("panda_arm")
    
    # 2. Set Reference Frame
    group.set_pose_reference_frame("panda_link0")
    
    # 3. Plan to Joint Goal (Safe)
    print("Moving to Ready...")
    # 'ready' is a named pose in SRDF
    group.set_named_target("ready") 
    success = group.go(wait=True)
    group.stop()
    
    # 4. Plan to Pose Goal
    print("Moving to Target Pose...")
    pose_goal = Pose()
    pose_goal.orientation.w = 1.0
    pose_goal.position.x = 0.4
    pose_goal.position.y = 0.1
    pose_goal.position.z = 0.4
    
    group.set_pose_target(pose_goal)
    
    # 5. Execute
    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    
    if success:
        print("Success!")
    else:
        print("Failed (Unreachable or Collision)")
        
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 👨‍💻 Adding Obstacles (`scripts/box_pick.py`)

```python
# ... Imports ...

def add_box(scene):
    pose = PoseStamped()
    pose.header.frame_id = "panda_link0"
    pose.pose.position.x = 0.5
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.2 # Table height
    
    # Add a box (Table)
    scene.add_box("table", pose, size=(0.5, 1.0, 0.4))
    
    # Add a target object
    pose.pose.position.z = 0.5 # On top of table
    scene.add_box("target", pose, size=(0.05, 0.05, 0.05))
    
    # Wait for sync
    time.sleep(2)

def main():
    # ... setup ...
    add_box(scene)
    
    # Plan
    group.set_position_target([0.5, 0.0, 0.6]) # Above target
    plan = group.plan()
    
    # MoveIt will avoid the "table" automatically
    group.execute(plan[1], wait=True)
```

---

## 🔬 Lab Exercise: The "Cage"

### 1. Lab Objectives
- Create a Planning Scene with the robot inside a cage (walls on 4 sides).
- **Task:** Plan to a point outside the cage (through the door).
- **Solver:** RRTConnect.
- **Observe:** The robot twists and folds itself to exit the door without touching walls.
- **Fail Case:** If the door is closed, planning fails after timeout.

---

## 🚀 Project: "Digital Twin Sync"

**Goal:** Run MoveIt on a PC, controlling a Simulation *and* visualization.
1.  **Fake Controller:** `ros2_control` hardware interface `mock_components/GenericSystem`.
2.  **Rviz:** Shows the ghost (Planned path) and the real robot (Joint States).
3.  **Visualization:** Enable "Motion Planning -> Planned Path -> Loop Animation" to see the preview.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Invalid Trajectory: Start Point Deviates"
*   **Symptom:** Planner fails immediately.
*   **Cause:** The robot moved *after* the plan was calculated, or the simulation has gravity sag. MoveIt requires `start_state` to match `current_state`.
*   **Fix:** Increase tolerance `allowed_start_tolerance`. Or plan and execute immediately.

#### 2. "Execution Failed: Trajectory Tolerance Violated"
*   **Symptom:** Robot moves but stops midway with error.
*   **Cause:** The PID controller on joints couldn't keep up with the trajectory (Too fast).
*   **Fix:** `group.set_max_velocity_scaling_factor(0.1)` (Slow down).

---

## ⚡ Optimization: Pilz Industrial Motion

For simple Pick & Place, RRT is overkill (random wiggle).
*   **Pilz Planner:** Generates deterministic LIN (Linear) and PTP (Point-To-Point) moves.
*   Like G-Code (CNC).
*   Use `group.set_planner_id("LIN")` for straight line insertions.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Planning Scene"?
    *   **A:** The 3D world representation (Robot + Obstacles) used for collision checking.
2.  **Q:** Difference between "Joint Goal" and "Pose Goal"?
    *   **A:** Joint Goal specifies angles (Unique). Pose Goal specifies XYZ/RPY (IK required, might have multiple solutions).
3.  **Q:** Why Time Parameterization?
    *   **A:** The planner outputs path (geometry). We need to assign timestamps ($t_0, t_1...$) such that vel/accel limits are respected.

### Challenge Task
> **Task:** Visualizing Reachability.
> 1. Sample 1000 random valid joint configs.
> 2. Plot the EE positions as a point cloud.
> 3. This cloud represents the **Workspace** volume.

---

## 📚 Further Reading
- **MoveIt 2 Tutorials:** The official documentation is excellent.
- **OMPL:** "Sampling-based algorithms for motion planning".

---

**Day 52 Complete**
