# Day 58: Physics Engines (Gazebo/Ignition)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 9: Simulation & Sim-to-Real

---

> **📝 Content Creator Instructions:**
> Code works on your laptop. But does it work on Mars?
> - **Focus:** Gazebo Classic vs Ignition (Gazebo Harmony), PyBullet (ML Training), and NVIDIA Isaac Sim (Photorealism).
> - **Code:** Spawning the `my_bot` from Day 57 into a Gazebo world with physics enabled.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Compare** physics engines (ODE, Bullet, PhysX) based on accuracy vs speed.
2.  **Launch** a ROS 2 Gazebo simulation and spawn a URDF entity.
3.  **Control** joint efforts/velocities via `ros2_control` within the simulation.
4.  **Debug** "Exploding Simulations" (NaN errors, constraint violations).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU recommended (OpenGL 3.3+).

### Software Environment
```bash
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-ros2-control
```

### Prior Knowledge
- URDF (Day 57).
- Launch Files (Day 5).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Landscape

1.  **Gazebo Classic (v11):** Old, stable, tightly integrated with ROS 1/2. Uses ODE (Open Dynamics Engine).
2.  **Gazebo (formerly Ignition):** New architecture. Modular. Better rendering (Ogre 2).
3.  **PyBullet:** Python-first. No ROS dependency required. extremely fast, preferred for Reinforcement Learning (RL).
4.  **Isaac Sim:** NVIDIA Omniverse. PhysX 5. Ray-tracing. Requires RTX GPU. Industry Standard for Digital Twins.

### 🔹 Part 2: Physics Parameters

*   **Time Step ($dt$):** usually 1ms (1000Hz).
*   **Solver Iterations:** More iterations = stiffer constraints (less jitter), but slower CPU.
*   **ERP (Error Reduction Parameter):** How fast to fix constraint errors (0=sponge, 1=rigid).
*   **CFM (Constraint Force Mixing):** Softness of constraints.

### 🔹 Part 3: `ros2_control` Interface

Gazebo is just physics. ROS is control.
*   **Plug-in:** `gazebo_ros2_control`. acts as the "Hardware Interface".
*   It reads `JointState` from Gazebo API $\to$ ROS Topic.
*   It takes `Command` from ROS Topic $\to$ Gazebo API (Apply Force).

---

## 💻 Implementation: Spawning `my_bot`

We use the URDF from Day 57.

### 🛠️ Project Structure
```text
day58_gazebo/
├── launch/
│   └── sim.launch.py
├── config/
│   └── controllers.yaml
├── world/
│   └── empty.world
└── urdf/
    └── gazebo_plugins.xacro (Add this to robot.urdf.xacro)
```

### 👨‍💻 Gazebo Plugin (`urdf/gazebo_plugins.xacro`)

Add this to your robot description to enable control.

```xml
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
    <gazebo>
        <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
            <parameters>$(find day58_gazebo)/config/controllers.yaml</parameters>
        </plugin>
    </gazebo>

    <ros2_control name="GazeboSystem" type="system">
        <hardware>
            <plugin>gazebo_ros2_control/GazeboSystem</plugin>
        </hardware>
        <joint name="left_wheel_joint">
            <command_interface name="velocity">
                <param name="min">-10</param>
                <param name="max">10</param>
            </command_interface>
            <state_interface name="position"/>
            <state_interface name="velocity"/>
        </joint>
        <joint name="right_wheel_joint">
            <command_interface name="velocity"/>
            <state_interface name="position"/>
            <state_interface name="velocity"/>
        </joint>
    </ros2_control>
</robot>
```

### 👨‍💻 Controllers Config (`config/controllers.yaml`)

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100 # Hz

    diff_drive_controller:
      type: diff_drive_controller/DiffDriveController

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

diff_drive_controller:
  ros__parameters:
    left_wheel_names: ["left_wheel_joint"]
    right_wheel_names: ["right_wheel_joint"]
    wheel_separation: 0.35
    wheel_radius: 0.05
```

### 👨‍💻 Launch File (`launch/sim.launch.py`)

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # 1. Start Gazebo Server & Client
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )
    
    # 2. Process URDF
    pkg_path = get_package_share_directory('day57_urdf')
    xacro_file = os.path.join(pkg_path, 'urdf', 'robot.urdf.xacro')
    robot_desc = xacro.process_file(xacro_file).toxml()
    
    # 3. Spawn Entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_bot'],
        output='screen'
    )
    
    # 4. Robot State Publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )
    
    # 5. Load Controllers (Delayed)
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )
    
    load_diff_drive_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller"],
    )

    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster, load_diff_drive_controller],
            )
        )
    ])
```

---

## 🔬 Lab Exercise: The "Galileo" Test

### 1. Lab Objectives
- Spawn two cubes at height $z=10m$.
- Use `libgazebo_ros_p3d` (Ground Truth) to log position.
- **Check 1:** Does it hit the ground in $t = \sqrt{2h/g} = \sqrt{20/9.81} \approx 1.42s$?
- **Check 2:** Change friction of ground to 0.0 (Ice). drive robot. Does it slide forever?
- **Goal:** Verify Simulation Physics parameters match Reality.

---

## 🚀 Project: "The Maze Runner"

**Goal:** Build a World file.
1.  **Editor:** Gazebo "Building Editor".
2.  **Walls:** Create a simple maze.
3.  **Spawn:** Place robot at Start.
4.  **Save:** `maze.world`.
5.  **Challenge:** Drive through it using keyboard teleop without flipping over (Physics instability).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot falls through floor"
*   **Symptom:** Subterranean robot.
*   **Cause:** No Collision geometry on the floor plane. Or Robot start height < 0.
*   **Fix:** Ensure `<include><uri>model://ground_plane</uri></include>` is in world file. Spawn at $z=0.1$.

#### 2. "Jittery Wheels"
*   **Symptom:** Robot vibrates when standing still.
*   **Cause:** PID gains in `gazebo_ros2_control` are too high. Or Mass of wheel is too small relative to Base (Mass Ratio > 1000:1 causes ODE numerical errors).
*   **Fix:** Increase wheel mass (dummy mass). Tune PID.

---

## ⚡ Optimization: Real-Time Factor (RTF)

*   **RTF = 1.0:** Simulation runs at real speed.
*   **RTF < 1.0:** Simulation is lagging (CPU overload). Lidar running at 5Hz instead of 10Hz.
*   **Fix:** Reduce Lidar rays. Use simplified collision meshes (Box instead of Mesh). Increase time step ($dt$) slightly.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why PyBullet for Reinforcement Learning?
    *   **A:** It creates 1000 headless environments in parallel. Faster training. Gazebo is single-instance (mostly).
2.  **Q:** What is a "Headless" simulation?
    *   **A:** No GUI (rendering). Just physics calculations. Much faster. Good for CI/CD testing.
3.  **Q:** Is Friction coefficient $\mu$ enough?
    *   **A:** No. Contact stiffness ($k_p$), damping ($k_d$), and slip compliance also matter for grasp stability.

### Challenge Task
> **Task:** Moon Gravity.
> 1. Set gravity to $-1.625 m/s^2$.
> 2. Drive the robot.
> 3. Observe how it "hops" on bumps. The suspension needs re-tuning.

---

## 📚 Further Reading
- **Gazebo Tutorials:** gazebosim.org.
- **Isaac Sim Docs:** Nvidia Omniverse.

---

**Day 58 Complete**
