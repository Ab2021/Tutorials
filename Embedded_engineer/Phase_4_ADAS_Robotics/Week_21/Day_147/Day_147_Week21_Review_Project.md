# Day 147: Week 21 Review & Project (The Final Capstone)
## Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone

---

> **📝 Day 147 Focus:**
> This is it. The culmination of 21 weeks of learning. We have built the eyes (Perception), the inner ear (Localization), the brain (Planning), and the hands (Control). Today, we assemble the **Complete Autonomous Driving Stack** and let it loose in the CARLA simulator.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** all subsystems into a unified ROS 2 architecture.
2.  **Launch** the full stack with a single command.
3.  **Monitor** system health and performance.
4.  **Execute** a complex mission (A to B in Urban Traffic).
5.  **Reflect** on the challenges of system integration.

---

## 📚 Week 21 Review

### 1. Lifecycle Management
-   Nodes have states (Unconfigured, Inactive, Active).
-   Manager orchestrates startup.

### 2. Launch & Config
-   Python launch files are powerful.
-   YAML params separate config from code.

### 3. Containerization
-   Docker ensures reproducibility.
-   CI/CD automates testing.

### 4. Simulation
-   CARLA provides the testbed.
-   Bridge connects ROS to Sim.

---

## 🛠️ Capstone Project: The "Autopilot" Stack

**Goal:** Navigate from Start Point to Goal Point in CARLA Town01, obeying traffic rules and avoiding obstacles.

**Architecture:**
1.  **Sensing:** CARLA Bridge (Lidar, Camera, GNSS).
2.  **Perception:** YOLO (2D Detect) + PointPillars (3D Detect).
3.  **Localization:** EKF Fusion (GNSS + IMU + Odom).
4.  **Planning:**
    -   Global: A* (Route).
    -   Behavior: FSM (Stop/Go).
    -   Local: MPC (Trajectory).
5.  **Control:** PID (Steering/Throttle).

### Package Structure
Create `week21_capstone` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week21_capstone
mkdir -p week21_capstone/launch
mkdir -p week21_capstone/config
```

### 👨‍💻 Code: The Master Launch (`launch/autopilot.launch.py`)

This launch file brings up the world.

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 1. CARLA Bridge (The World)
    carla_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('week21_day145'), 'launch', 'carla_bridge.launch.py')
        )
    )
    
    # 2. Perception Stack (The Eyes)
    # Assumes you have a perception launch file from Week 18
    # perception_launch = ...
    
    # 3. Localization Stack (The Pose)
    # Assumes you have a localization launch file from Week 19
    # localization_launch = ...
    
    # 4. Planning & Control (The Brain)
    # Assumes you have a planning launch file from Week 20
    # planning_launch = ...
    
    # 5. Rviz (The Visualization)
    rviz_config = os.path.join(get_package_share_directory('week21_capstone'), 'config', 'capstone.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        carla_launch,
        # TimerAction(period=5.0, actions=[perception_launch]), # Wait for sim
        # TimerAction(period=7.0, actions=[localization_launch]),
        # TimerAction(period=10.0, actions=[planning_launch]),
        rviz_node
    ])
```

### 👨‍💻 Code: The Autopilot Node (`autopilot_node.py`)

A simplified "glue" node that takes Perception/Localization and outputs Control.

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from carla_msgs.msg import CarlaEgoVehicleControl
from visualization_msgs.msg import MarkerArray

class Autopilot(Node):
    def __init__(self):
        super().__init__('autopilot')
        
        # Subs
        self.sub_odom = self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.odom_cb, 10)
        self.sub_objects = self.create_subscription(MarkerArray, '/perception/objects', self.obj_cb, 10)
        
        # Pubs
        self.pub_ctrl = self.create_publisher(CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        
        # State
        self.current_pose = None
        self.objects = []
        self.target_speed = 20.0 # km/h
        
        # Timer (Control Loop 20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

    def odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def obj_cb(self, msg):
        self.objects = msg.markers

    def control_loop(self):
        if not self.current_pose: return
        
        # 1. Simple Behavior Logic
        # If object ahead < 10m, Stop.
        stop = False
        for obj in self.objects:
            # Calculate distance (Simplified)
            dist = ((obj.pose.position.x - self.current_pose.position.x)**2 + 
                    (obj.pose.position.y - self.current_pose.position.y)**2)**0.5
            if dist < 10.0:
                stop = True
                break
        
        # 2. Control Output
        cmd = CarlaEgoVehicleControl()
        if stop:
            cmd.throttle = 0.0
            cmd.brake = 1.0
            self.get_logger().info("Stopping!")
        else:
            cmd.throttle = 0.5
            cmd.brake = 0.0
            cmd.steer = 0.0 # Drive straight for demo
            
        self.pub_ctrl.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = Autopilot()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 🧪 Verification & Testing

### 1. The "Hello World" Drive
-   Launch the stack.
-   Car should spawn.
-   Car should accelerate.
-   Car should stop if you spawn an obstacle in front of it (using CARLA Python API).

### 2. The Loop
-   Set a goal waypoint around the block.
-   Verify the A* planner generates a path.
-   Verify the Controller follows the path.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Integration
1.  **Q:** Why do we use `TimerAction` in the launch file?
    *   **A:** To stagger the startup. Starting everything at once causes CPU spikes and race conditions (e.g., Planner starts before Map is ready).
2.  **Q:** How do we debug if the car is oscillating?
    *   **A:** Check the Controller (PID gains) and the Latency (Day 146). High latency causes oscillation.

### Section 2: The Big Picture
3.  **Q:** What is the hardest part of autonomous driving?
    *   **A:** The "Long Tail" of edge cases. The first 90% is easy. The last 1% (snow, construction, erratic humans) is 99% of the work.

---

## 🏆 Conclusion

**Congratulations!** You have completed Phase 4 of the Embedded Engineer Course.
You have gone from blinking an LED (Phase 1) to building a self-driving car stack (Phase 4).

**What's Next?**
-   **Phase 5:** Advanced Specialization (Optional).
-   **Career:** You are now ready for roles like "Robotics Software Engineer", "ADAS Engineer", or "Perception Engineer".

**Keep Building.** The road is long, but you have the map.

---

**Day 147 Complete** | Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone
