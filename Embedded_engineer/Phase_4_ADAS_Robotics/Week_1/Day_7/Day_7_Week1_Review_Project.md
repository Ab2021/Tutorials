# Day 7: Week 1 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 1: ROS 2 Fundamentals

---

> **📝 Day 7 Focus:**
> We have covered the foundational building blocks of ROS 2: Architecture, Communication Patterns (Topics, Services, Actions), Configuration (Launch, Params), and Geometry (TF2, URDF). Today, we synthesize this knowledge into a cohesive **Week 1 Capstone Project**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Synthesize** multiple ROS 2 concepts (Nodes, TF2, Launch, URDF) into a single working application.
2.  **Architect** a modular ROS 2 package structure suitable for production.
3.  **Implement** a "Virtual ADAS Platform" that simulates sensors, transforms data, and fuses it.
4.  **Debug** complex system interactions using the full suite of ROS 2 tools.
5.  **Evaluate** your understanding of Week 1 concepts through a comprehensive assessment.

---

## 📚 Week 1 Review

### 1. Architecture & Communication
-   **Nodes:** The executable units. Use `LifecycleNodes` for safety.
-   **Executors:** `MultiThreadedExecutor` + `CallbackGroups` for concurrency.
-   **Topics:** Streaming data (Sensors). Use QoS for reliability/latency trade-offs.
-   **Services:** Request/Response (State queries). Synchronous calls block!
-   **Actions:** Long-running tasks (Navigation). Goal/Feedback/Result.

### 2. Configuration & Launch
-   **Launch:** Python-based orchestration. Use events (`OnProcessExit`) for robustness.
-   **Parameters:** Runtime configuration. Use `add_on_set_parameters_callback`.
-   **YAML:** Bulk loading of parameters.

### 3. Geometry & Transforms
-   **TF2:** The tree of coordinate frames.
-   **Static Transforms:** Fixed mounting points (`/tf_static`).
-   **Dynamic Transforms:** Moving joints/vehicles (`/tf`).
-   **URDF/Xacro:** The physical description of the robot.

---

## 🛠️ Capstone Project: Virtual ADAS Platform

**Goal:** Build a complete ROS 2 system that simulates an autonomous vehicle with a rotating LiDAR and a fixed Camera. The system must:
1.  Load a URDF model of the vehicle.
2.  Publish static transforms for the camera.
3.  Publish dynamic transforms for the rotating LiDAR.
4.  Simulate sensor data (PointClouds and Images).
5.  Fuse the data in a central node (transforming LiDAR points to Camera frame).
6.  Launch everything with a single command.

### Package Structure
We will create a meta-package or a single large package `adas_week1_project`.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake \
  --dependencies rclcpp std_msgs sensor_msgs geometry_msgs tf2_ros tf2_geometry_msgs urdf xacro \
  --node-name fusion_node \
  adas_week1_project

cd adas_week1_project
mkdir launch urdf rviz config
```

### Step 1: The Vehicle Model (URDF)

Create `urdf/vehicle.xacro`:

```xml
<?xml version="1.0"?>
<robot name="adas_bot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Chassis -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="2.0 1.0 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
  </link>

  <!-- Camera Link (Fixed) -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="1.0 0 0.5" rpy="0 0 0"/>
  </joint>

  <!-- LiDAR Link (Rotating) -->
  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="lidar_joint" type="continuous">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

</robot>
```

### Step 2: The Sensor Simulator Node

This node will:
1.  Publish `sensor_msgs/JointState` to rotate the LiDAR.
2.  Publish dummy `sensor_msgs/PointCloud2` in `lidar_link`.
3.  Publish dummy `sensor_msgs/Image` in `camera_link`.

Create `src/sensor_sim.cpp`:

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cmath>

class SensorSim : public rclcpp::Node
{
public:
  SensorSim() : Node("sensor_sim")
  {
    // Publishers
    joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    lidar_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar/points", 10);
    camera_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image", 10);

    // Timer (20Hz)
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&SensorSim::timer_callback, this));
      
    start_time_ = this->now();
  }

private:
  void timer_callback()
  {
    auto now = this->now();
    double elapsed = (now - start_time_).seconds();

    // 1. Publish Joint State (Rotate LiDAR)
    sensor_msgs::msg::JointState joint_msg;
    joint_msg.header.stamp = now;
    joint_msg.name.push_back("lidar_joint");
    joint_msg.position.push_back(elapsed * 1.0); // 1 rad/s
    joint_pub_->publish(joint_msg);

    // 2. Publish Dummy PointCloud
    // We'll create a single point that moves in and out
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = now;
    cloud_msg.header.frame_id = "lidar_link";
    cloud_msg.height = 1;
    cloud_msg.width = 1;
    cloud_msg.fields.resize(3);
    cloud_msg.fields[0].name = "x"; cloud_msg.fields[0].offset = 0; cloud_msg.fields[0].datatype = 7; cloud_msg.fields[0].count = 1;
    cloud_msg.fields[1].name = "y"; cloud_msg.fields[1].offset = 4; cloud_msg.fields[1].datatype = 7; cloud_msg.fields[1].count = 1;
    cloud_msg.fields[2].name = "z"; cloud_msg.fields[2].offset = 8; cloud_msg.fields[2].datatype = 7; cloud_msg.fields[2].count = 1;
    cloud_msg.point_step = 12;
    cloud_msg.row_step = 12;
    cloud_msg.is_bigendian = false;
    cloud_msg.is_dense = true;
    
    // Manual serialization of float32
    cloud_msg.data.resize(12);
    float x = 5.0 + std::sin(elapsed);
    float y = 0.0;
    float z = 0.0;
    memcpy(&cloud_msg.data[0], &x, 4);
    memcpy(&cloud_msg.data[4], &y, 4);
    memcpy(&cloud_msg.data[8], &z, 4);
    
    lidar_pub_->publish(cloud_msg);

    // 3. Publish Dummy Image (Empty)
    sensor_msgs::msg::Image img_msg;
    img_msg.header.stamp = now;
    img_msg.header.frame_id = "camera_link";
    img_msg.height = 480;
    img_msg.width = 640;
    img_msg.encoding = "rgb8";
    camera_pub_->publish(img_msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time start_time_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorSim>());
  rclcpp::shutdown();
  return 0;
}
```

### Step 3: The Fusion Node (TF Listener)

This node will:
1.  Listen to `lidar/points`.
2.  Transform the points from `lidar_link` to `camera_link`.
3.  Publish the transformed points.

Create `src/fusion_node.cpp`:

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp> // For doTransform

class FusionNode : public rclcpp::Node
{
public:
  FusionNode() : Node("fusion_node")
  {
    // Buffer & Listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Subscriber
    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "lidar/points", 10,
      std::bind(&FusionNode::lidar_callback, this, std::placeholders::_1));

    // Publisher
    fused_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("fused/points", 10);
  }

private:
  void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    try {
      // Wait for transform from lidar_link to camera_link
      // We use the timestamp from the message
      if (tf_buffer_->canTransform("camera_link", msg->header.frame_id, 
                                   msg->header.stamp, rclcpp::Duration::from_seconds(0.1))) 
      {
        sensor_msgs::msg::PointCloud2 transformed_cloud;
        tf2::doTransform(*msg, transformed_cloud, 
                         tf_buffer_->lookupTransform("camera_link", msg->header.frame_id, msg->header.stamp));
        
        fused_pub_->publish(transformed_cloud);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
          "Transformed cloud to camera frame");
      }
      else {
        RCLCPP_WARN(this->get_logger(), "Transform not available yet");
      }
    } catch (tf2::TransformException & ex) {
      RCLCPP_ERROR(this->get_logger(), "TF Error: %s", ex.what());
    }
  }

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fused_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FusionNode>());
  rclcpp::shutdown();
  return 0;
}
```

### Step 4: Launch File

Create `launch/system.launch.py`:

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('adas_week1_project')
    xacro_file = os.path.join(pkg_share, 'urdf', 'vehicle.xacro')
    rviz_config = os.path.join(pkg_share, 'config', 'view.rviz')

    # Robot State Publisher (Publishes TF from URDF + Joint States)
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', xacro_file])}]
    )

    # Sensor Simulator (Publishes Joint States + Data)
    sim_node = Node(
        package='adas_week1_project',
        executable='sensor_sim',
        name='sensor_sim'
    )

    # Fusion Node (Consumes Data + TF)
    fusion_node = Node(
        package='adas_week1_project',
        executable='fusion_node',
        name='fusion_node'
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        rsp_node,
        sim_node,
        fusion_node,
        rviz_node
    ])
```

### Step 5: CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(adas_week1_project)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

add_executable(sensor_sim src/sensor_sim.cpp)
ament_target_dependencies(sensor_sim rclcpp sensor_msgs)

add_executable(fusion_node src/fusion_node.cpp)
ament_target_dependencies(fusion_node rclcpp sensor_msgs tf2_ros tf2_sensor_msgs)

install(TARGETS sensor_sim fusion_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch urdf config
  DESTINATION share/${PROJECT_NAME})

ament_package()
```

---

## 🧪 Verification & Testing

### 1. Build
```bash
colcon build --packages-select adas_week1_project
source install/setup.bash
```

### 2. Run
```bash
ros2 launch adas_week1_project system.launch.py
```

### 3. RViz Setup
1.  **Fixed Frame:** `base_link`.
2.  **Add RobotModel:** See the vehicle.
3.  **Add TF:** See `lidar_link` rotating.
4.  **Add PointCloud2:** Topic `/lidar/points`. See the point moving.
5.  **Add PointCloud2:** Topic `/fused/points`. Change color.
    -   *Observation:* The fused point should appear in the same physical location as the lidar point, but its coordinates are transformed to the camera frame.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Architecture
1.  **Q:** Why is `rclcpp::spin()` necessary?
    *   **A:** It gives the Executor control to process callbacks (timers, subscriptions) from the event queue. Without it, the node sits idle.
2.  **Q:** What is the difference between `rclcpp::Node` and `rclcpp_lifecycle::LifecycleNode`?
    *   **A:** Lifecycle nodes have managed states (Unconfigured, Inactive, Active) and transition callbacks, allowing deterministic startup/shutdown.

### Section 2: Communication
3.  **Q:** You have a LiDAR publishing at 10Hz and a Camera at 30Hz. How do you process them together?
    *   **A:** Use `message_filters::TimeSynchronizer` (exact time) or `ApproximateTimeSynchronizer` to callback only when both messages arrive with close timestamps.
4.  **Q:** Why might a Service Client hang indefinitely?
    *   **A:** Server is down, or Client is calling synchronously from a callback that blocks the Executor (Deadlock).

### Section 3: TF2 & URDF
5.  **Q:** If `robot_state_publisher` is running, why do I need `joint_state_publisher` (or a simulator)?
    *   **A:** RSP needs the *values* of the joints (angles) to calculate the transforms. It knows the geometry (URDF) but not the current state.
6.  **Q:** What is the parent frame of `base_link` usually?
    *   **A:** `odom` (if moving) or `map` (if localized).

### Section 4: Launch
7.  **Q:** How do you make a node respawn if it crashes?
    *   **A:** Use `respawn=True` in the `Node` action (Humble+), or an `OnProcessExit` event handler.

---

## 🏆 Conclusion

Congratulations on completing Week 1! You now have a solid foundation in ROS 2.
-   You can build nodes, define interfaces, and manage configurations.
-   You understand the coordinate systems that underpin all robotics.
-   You have built a working simulation of a multi-sensor vehicle.

**Next Week:** We dive into **Sensor Fusion & Perception**. We will take the simulated data and replace it with real algorithms (Kalman Filters) and process real datasets (KITTI).

---

**Day 7 Complete** | Phase 4: ADAS & Robotics Systems | Week 1: ROS 2 Fundamentals
