# Day 167: System Integration & Simulation
## Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking

---

> **📝 Day 167 Focus:**
> We have the parts: The Map (Day 163), The Planner (Day 164), The Eyes (Day 165), and The Hands (Day 166). Today, we assemble the robot. We integrate the full **AVP Stack** in ROS 2 and run it in **CARLA**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Architect** the AVP State Machine (Mission Control).
2.  **Launch** the full stack (Perception, Localization, Planning, Control).
3.  **Execute** a complete parking mission in CARLA.
4.  **Debug** integration issues (Latency, TF errors).
5.  **Validate** the final parking accuracy.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **All Previous Days.**

### Hardware Requirements
-   **GPU:** For CARLA.

### Software Stack
-   **ROS 2:** `nav2_behavior_tree` (Optional) or Python State Machine.
-   **CARLA:** Town04 (Parking Garage).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Mission Logic

The "Brain" that coordinates the modules.
1.  **IDLE:** Wait for user command ("Park Here").
2.  **MAPPING:** Drive to the drop-off zone (Localization init).
3.  **SEARCHING:** Drive through aisles, looking for empty slots (Perception).
4.  **PLANNING:** Found slot. Stop. Compute Hybrid A* path.
5.  **PARKING:** Execute path (MPC). Handle gear shifts.
6.  **PARKED:** Secure vehicle (Handbrake).

### 🔹 Part 2: Integration Challenges

-   **TF Tree:** The Map frame must align with the CARLA world.
-   **Time Sync:** Lidar and Camera must be synced to avoid "smearing" obstacles.
-   **Latency:** If Perception takes 200ms, the Controller is reacting to old news.

---

## 💻 Implementation: The AVP State Machine

**Scenario:**
-   Full Mission Control node.

### 🛠️ Setup
Update `week24_capstone`.

### 👨‍💻 Code: Mission Control Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path

class AVPMissionControl(Node):
    def __init__(self):
        super().__init__('avp_mission_control')
        
        # State
        self.state = "IDLE"
        self.target_slot = None
        
        # Pubs/Subs
        self.pub_state = self.create_publisher(String, '/system/state', 10)
        self.pub_goal = self.create_publisher(PoseStamped, '/planning/goal', 10)
        
        self.create_subscription(Bool, '/user/park_cmd', self.cmd_cb, 10)
        self.create_subscription(PoseStamped, '/perception/empty_slot', self.slot_cb, 10)
        self.create_subscription(Bool, '/control/finished', self.finished_cb, 10)
        
        self.timer = self.create_timer(0.1, self.loop)
        
    def cmd_cb(self, msg):
        if msg.data and self.state == "IDLE":
            self.state = "SEARCHING"
            self.get_logger().info("Mission Start: SEARCHING")

    def slot_cb(self, msg):
        if self.state == "SEARCHING":
            self.target_slot = msg
            self.state = "PLANNING"
            self.get_logger().info(f"Slot Found at {msg.pose.position.x:.1f}, {msg.pose.position.y:.1f}")
            
            # Send Goal to Planner
            self.pub_goal.publish(msg)

    def finished_cb(self, msg):
        if msg.data and self.state == "PARKING":
            self.state = "PARKED"
            self.get_logger().info("Mission Complete: PARKED")

    def loop(self):
        # Publish State
        msg = String()
        msg.data = self.state
        self.pub_state.publish(msg)
        
        # Logic
        if self.state == "PLANNING":
            # Wait for path... (Simplified)
            # In real code, subscribe to /planning/path
            self.state = "PARKING"
            self.get_logger().info("Path Planned. Executing...")

def main():
    rclpy.init()
    node = AVPMissionControl()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

### 👨‍💻 Code: The Launch File (`launch/avp.launch.py`)

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Map Server
        Node(package='week24_capstone', executable='map_server', name='map_server'),
        
        # 2. Perception (Mock or Real)
        Node(package='week24_capstone', executable='perception_node', name='perception'),
        
        # 3. Planner
        Node(package='week24_capstone', executable='planning_node', name='planner'),
        
        # 4. Controller
        Node(package='week24_capstone', executable='control_node', name='controller'),
        
        # 5. Mission Control
        Node(package='week24_capstone', executable='mission_control', name='mission_control'),
        
        # 6. Rviz
        Node(package='rviz2', executable='rviz2', arguments=['-d', 'avp.rviz']),
    ])
```

---

## 🔬 Lab Exercise: The Grand Finale

### Lab Objectives
1.  **Start CARLA:**
    ```bash
    ./CarlaUE4.sh
    ```
2.  **Start Bridge:**
    ```bash
    ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py
    ```
3.  **Start AVP Stack:**
    ```bash
    ros2 launch week24_capstone avp.launch.py
    ```
4.  **Trigger:**
    ```bash
    ros2 topic pub /user/park_cmd std_msgs/msg/Bool "data: true" -1
    ```
5.  **Watch:**
    -   The car drives down the aisle.
    -   It detects a spot (Green box in Rviz).
    -   It stops.
    -   It reverses into the spot.
    -   "Mission Complete".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Car Hits the Wall
**Symptom:** MPC tracks perfectly, but the car hits the pillar.
**Cause:** **Vehicle Dimensions**. The Planner assumed the car was 1.8m wide, but the mirrors make it 2.1m.
**Solution:** Inflate the vehicle footprint in the Planner (Safety Margin).

#### 2. Localization Jump
**Symptom:** Car teleports 1m sideways in Rviz.
**Cause:** SLAM loop closure or GPS multipath.
**Solution:** Trust Odometry/IMU more during the parking maneuver (Short term). Ignore GPS jumps.

---

## ⚡ Optimization & Best Practices

### 1. Behavior Trees (BT)
-   Instead of `if-else` State Machine, use **Behavior Trees** (`nav2_behavior_tree`).
-   Better for complex logic: "If Slot Found -> Sequence(Stop, Blinkers, Reverse)".
-   Easier to recover from failures ("If Path Blocked -> Wait -> Re-plan").

### 2. Safety Monitor
-   Run a separate node that checks **Time-to-Collision (TTC)**.
-   If TTC < 1s, override *everything* and E-Stop.
-   This runs at 100Hz on the microcontroller (Safety Island), not the main PC.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Handover" phase?
    *   **A:** When the driver leaves the car and transfers control to the AVP system.
2.  **Q:** How does the car know it is parked?
    *   **A:** Pose error < threshold AND Velocity = 0 AND Gear = Park.
3.  **Q:** Why use a State Machine?
    *   **A:** To enforce a strict sequence of operations and prevent undefined behavior (e.g., trying to park while driving on the highway).

### Challenge Task
**Task:** Obstacle Injection.
1.  While the car is reversing, throw a virtual pedestrian behind it.
2.  Verify the Safety Monitor triggers E-Stop.
3.  Verify the Mission Control handles the interruption (Wait vs Abort).

---

## 📚 Further Reading & References
-   [Autoware AVP Demo](https://www.youtube.com/watch?v=...)
-   [Behavior Trees in Robotics](https://arxiv.org/abs/1709.00084)

---

**Day 167 Complete** | Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking
