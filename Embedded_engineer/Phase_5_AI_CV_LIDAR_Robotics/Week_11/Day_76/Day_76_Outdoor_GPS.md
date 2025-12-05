# Day 76: Outdoor GPS Navigation
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 11: Navigation 2 (Nav2) Mastery

---

> **📝 Content Creator Instructions:**
> No walls. No Map? Just Lat/Lon.
> - **Focus:** GPS Integration, `navsat_transform_node`, `robot_localization` (Dual EKF), and Nav2 GPS Waypoint Following.
> - **Code:** A `gps_waypoint_follower` node that accepts `NavSatFix` goals and converts them to `map` frame coordinates for Nav2.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** `robot_localization` (RL) for GPS Fusion (Dual EKF setup).
2.  **Explain** the role of `navsat_transform_node`: WGS84 (Lat/Lon) $\to$ UTM $\to$ Odom/Map.
3.  **Execute** GPS Waypoint Following behaviors.
4.  **Handle** GPS Dropouts: What happens when the robot goes under a tree?

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPS Module (U-Blox F9P recommended for RTK).
- IMU (Magnetometer needed for Absolute Heading).

### Software Environment
```bash
sudo apt install ros-humble-robot-localization
pip install geopy
```

### Prior Knowledge
- EKF (Day 10).
- TF Trees (earth -> map -> odom -> base_link).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Coordinate Crisis

*   **GPS:** Spherical (Lat, Lon, Alt). Frame: `earth`.
*   **Robot:** Cartesian (X, Y in meters). Frame: `map` or `odom`.
*   **Solution: UTM (Universal Transverse Mercator).**
    *   Projects Earth slice to a flat 2D grid.
    *   `navsat_transform_node` handles this conversion + datum origin.

### 🔹 Part 2: Dual EKF Architecture

We run TWO instances of EKF:
1.  **Local EKF (Odom Frame):**
    *   Fuses: Wheel Encoders + IMU (Gyro/Accel).
    *   Smooth. Continuous. Drifts over time.
    *   Used for: `cmd_vel` control loops (High frequency).
2.  **Global EKF (Map Frame):**
    *   Fuses: Wheel Encoders + IMU + **GPS**.
    *   Discrete jumps (GPS jumps). Bounded Global Error.
    *   Used for: Global Planning (`map` to `base_link` tf).

### 🔹 Part 3: Nav2 with No Map?

*   **SLAM:** Needed if you want to avoid obstacles.
*   **GPS-Only:** If field is open, you don't need a static map.
*   **Hybrid:** Use a "Rolling Window" local costmap (50x50m) centered on the robot, but navigate to Global GPS coordinates.

---

## 💻 Implementation: GPS Waypoint Follower

ROS 2 Actions typically use `PoseStamped` (Meters). We need `GeoPose` (Lat/Lon).

### 🛠️ Project Structure
```text
day76_gps/
├── config/
│   └── dual_ekf_navsat.yaml
├── src/
│   └── gps_commander.py
└── launch/
    └── gps_nav.launch.py
```

### 👨‍💻 EKF Configuration (`config/dual_ekf_navsat.yaml`)

```yaml
ekf_filter_node_map:
  ros__parameters:
    frequency: 30.0
    sensor_timeout: 0.1
    two_d_mode: true
    publish_tf: true
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link
    world_frame: map

    # Input 1: Odom
    odom0: /odometry/filtered/local # Output of Local EKF
    odom0_config: [true, true, false, ... ] 

    # Input 2: GPS (As Odometry via navsat_transform)
    odom1: /odometry/gps
    odom1_config: [true, true, false, ... ] # X, Y, Z
    odom1_differential: false # GPS is Absolute

navsat_transform:
  ros__parameters:
    frequency: 30.0
    magnetic_declination_radians: 0.0 # Check your city!
    yaw_offset: 0.0 # IMU mounting
    zero_altitude: true
    broadcast_cartesian_transform: true # Publishes map->odom? No, utm->map
    publish_filtered_gps: true
    use_odometry_yaw: false
    wait_for_datum: false
```

### 👨‍💻 GPS Commander Node (`src/gps_commander.py`)

Converts Lat/Lon to `map` frame using `robot_localization` service `fromLL`.

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from robot_localization.srv import FromLL
from geographic_msgs.msg import GeoPoint

class GPSCommander(Node):
    def __init__(self):
        super().__init__('gps_commander')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.ll_client = self.create_client(FromLL, '/fromLL')
        
        while not self.ll_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /fromLL service...')

    def go_to_gps(self, lat, lon):
        # 1. Convert Lat/Lon to Map Frame Point
        req = FromLL.Request()
        req.ll_point = GeoPoint(latitude=lat, longitude=lon, altitude=0.0)
        
        future = self.ll_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        
        target_point = resp.map_point
        self.get_logger().info(f"Going to Map Coords: {target_point}")

        # 2. Send Nav2 Goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position = target_point
        goal_msg.pose.pose.orientation.w = 1.0 # Orientation doesn't matter for waypoint
        
        self.nav_client.call_async(goal_msg)

def main():
    rclpy.init()
    node = GPSCommander()
    # Example: Go to Eiffel Tower (Simulation)
    node.go_to_gps(48.8584, 2.2945) 
    rclpy.spin(node)
```

---

## 🔬 Lab Exercise: "The Datum"

### 1. Lab Objectives
- Use Gazebo `hector_gazebo_plugins` GPS sensor.
- Set the `datum` parameter in `navsat_transform_node`.
    - `datum: [Lat, Lon, Heading]`
    - This defines the $(0,0)$ of the `map` frame.
- **Task:** Drive robot 10 meters North.
- **Check:** `ros2 topic echo /gps/fix`. Latitude should increase.
- **Check:** `ros2 topic echo /odometry/gps`. Y should increase by ~10m.

---

## 🚀 Project: "Agricultural Rover"

**Goal:** Auto-Farming.
1.  **Input:** A list of Lat/Lon coordinates (Crop Rows).
2.  **Environment:** Open field (empty map).
3.  **Config:**
    *   `global_costmap`: Large size (200x200m), Static Layer = Disabled (or empty).
    *   `local_costmap`: Rolling Window (10x10m), Obstacle Layer (Lidar).
4.  **Behavior:** Navigate through points. Detect "Weeds" (Visual) $\to$ Stop & Spray.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Toilet Bowl Effect" (Circling)
*   **Symptom:** Robot drives in circles.
*   **Cause:** Magnetometer calibration is wrong OR `magnetic_declination` not set. The robot thinks North is East. It turns left to correct, but absolute GPS says it's going adjusting wrong.
*   **Fix:** Calibrate Magnetometer. Check `imu/data` orientation logic (ENU vs NED).

#### 2. "Flying Robot" (Z drift)
*   **Symptom:** Odom Z coordinate drifts to 100m.
*   **Cause:** GPS altitude noise.
*   **Fix:** Set `two_d_mode: true` in EKF. It forces $Z, Roll, Pitch$ to 0.

---

## ⚡ Optimization: RTK-GPS

Standard GPS has 2-5m accuracy. Not enough for sidewalk robots.
*   **RTK (Real-Time Kinematic):** Uses a fixed Base Station.
*   **NTRIP:** Receive correction data over Internet (4G).
*   **Accuracy:** 2cm.
*   **ROS Driver:** `ublox_dgnss` or `microstrain_inertial`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the `map` frame in GPS context?
    *   **A:** It is a local Tangent Plane attached to the Datum on Earth's surface.
2.  **Q:** Why do we need `navsat_transform`?
    *   **A:** EKF filters don't understand Spherical Coordinates. They need Linear/Cartesian inputs (Velocities/XY positions).
3.  **Q:** Can we use AMCL and GPS together?
    *   **A:** Yes. AMCL localizes in a Map. GPS corrects global drift. But usually redundant. Usually select ONE Source of Global Truth.

### Challenge Task
> **Task:** GPS Denied Zone.
> 1. Simulate entering a Tunnel.
> 2. GPS fix is lost (`status = -1`).
> 3. `robot_localization` should rely purely on IMU/Odom.
> 4. Covariance (Uncertainty) grows.
> 5. Exiting Tunnel: GPS returns. Filter "Converges" (Jumps back to truth).

---

## 📚 Further Reading
- **Robot Localization Docs:** "Integrating GPS".
- **ROS 2 Nav2:** "GPS Waypoint Follower Tutorial".

---

**Day 76 Complete**
