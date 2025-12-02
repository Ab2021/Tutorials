# Day 2: Nodes, Topics, and Publishers/Subscribers
## Phase 4: ADAS & Robotics Systems | Week 1: ROS 2 Fundamentals

---

> **📝 Day 2 Focus:**
> Deep dive into the fundamental building blocks of ROS 2: Nodes, Topics, and the Publish-Subscribe communication pattern. We will explore the internal architecture of nodes, the mechanics of topic communication, and how to implement robust publishers and subscribers for ADAS applications.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Deconstruct** the internal architecture of a ROS 2 Node, including Contexts, Executors, and Callback Groups.
2.  **Analyze** the Topic communication mechanism, including Interface Definition Language (IDL), Serialization (CDR), and the RMW layer.
3.  **Implement** robust, thread-safe Publishers and Subscribers in C++ using modern best practices.
4.  **Design** custom message interfaces (.msg) for specific ADAS sensor data.
5.  **Debug** communication issues using command-line tools and introspection.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **C++14/17:** Lambda functions, smart pointers (`std::shared_ptr`, `std::unique_ptr`), threading basics.
-   **Build Systems:** CMake fundamentals.
-   **ROS 2 Concepts:** Basic understanding from Day 1 (DDS, Architecture).

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS with ROS 2 Humble.
-   **Sensors (Optional):** USB Camera or simulated sensor data.

### Software Stack
-   **ROS 2 Humble**
-   **Colcon:** Build tool.
-   **Rosdep:** Dependency manager.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The ROS 2 Node Architecture

#### 1.1 What is a Node? (Beyond the Basics)

In ROS 1, a node was essentially a process. In ROS 2, the concept of a node is decoupled from the process execution model. A "Node" is an object within a process that serves as an entry point to the ROS 2 graph.

**Key Characteristics:**
-   **Process-Agnostic:** Multiple nodes can run in a single process (Component-based architecture).
-   **Context-Aware:** Nodes operate within a `rclcpp::Context`, which manages global state (DDS participants, signal handlers).
-   **Lifecycle-Managed:** Nodes can have managed states (Unconfigured, Inactive, Active, Finalized) - covered in Day 1, but relevant here for architecture.

**Internal Structure of a `rclcpp::Node`:**

1.  **Node Options (`rclcpp::NodeOptions`):**
    -   Configuration passed at construction.
    -   Controls parameter overrides, context usage, and argument parsing.
    -   *Critical for Component Containers.*

2.  **Interfaces:**
    -   **Parameters Interface:** Manages runtime configuration.
    -   **Graph Interface:** Provides information about the ROS graph (other nodes, topics).
    -   **Logging Interface:** Access to `rcutils` logging.
    -   **Clock Interface:** Manages time (System time, ROS time, Steady time).
    -   **Time Source:** Handles `/clock` simulation time.

3.  **Callback Groups:**
    -   Mechanism to control concurrent execution of callbacks.
    -   *Crucial for avoiding deadlocks in complex ADAS nodes.*

#### 1.2 Executors and Scheduling

The **Executor** is the engine that runs your node. It is responsible for taking incoming events (messages, timer ticks, service requests) and executing the corresponding callbacks.

**How it works:**
1.  **Wait:** The Executor waits for work (using the DDS WaitSet mechanism).
2.  **Fetch:** When work arrives, it fetches the corresponding callback.
3.  **Execute:** It runs the callback.

**Types of Executors:**

1.  **SingleThreadedExecutor (Default):**
    -   Runs all callbacks in a single thread.
    -   **Pros:** Thread-safe by default (no race conditions between callbacks).
    -   **Cons:** Blocking. If one callback takes too long (e.g., heavy image processing), it blocks all others (e.g., emergency stop signal).
    -   *Risk:* Head-of-line blocking.

2.  **MultiThreadedExecutor:**
    -   Creates a pool of threads.
    -   Can run multiple callbacks in parallel.
    -   **Pros:** High throughput, non-blocking.
    -   **Cons:** Requires thread safety (mutexes) for shared data.
    -   *Risk:* Race conditions, deadlocks.

3.  **StaticSingleThreadedExecutor:**
    -   Optimized for systems with fixed node structure.
    -   Scans for work only once (or less frequently).
    -   Lower CPU overhead.

**Callback Groups & Threading Models:**

When using a `MultiThreadedExecutor`, how do you control concurrency? **Callback Groups**.

-   **Mutually Exclusive Group (Default):**
    -   Callbacks in this group cannot run concurrently *with each other*.
    -   They can run concurrently with callbacks in *other* groups.
    -   *Use case:* Protecting shared state without explicit mutexes.

-   **Reentrant Group:**
    -   Callbacks in this group can run concurrently with *each other* (and themselves, if triggered multiple times).
    -   *Use case:* Stateless callbacks, or fully thread-safe handlers (e.g., pure coordinate transforms).

**ADAS Scenario:**
> You have a LiDAR processing callback (heavy) and an Emergency Stop callback (light, critical).
> -   **Bad:** Both in SingleThreadedExecutor. LiDAR blocks E-Stop.
> -   **Better:** MultiThreadedExecutor.
> -   **Best:** MultiThreadedExecutor with E-Stop in a separate Mutually Exclusive Group (or Reentrant) to ensure it can preempt or run alongside LiDAR processing.

#### 1.3 The Context (`rclcpp::Context`)

The Context is the "global" state of your ROS 2 application instance.

-   Initializes the middleware (DDS).
-   Manages the `shutdown` signal.
-   Allows multiple ROS 2 "instances" in one process (rare but possible).
-   **Global vs. Local:** Usually, we use the default global context (`rclcpp::init`), but for testing or isolation, custom contexts are used.

---

### 🔹 Part 2: Topics & Communication Mechanics

#### 2.1 The Publish-Subscribe Pattern

Topics implement a **many-to-many**, **asynchronous** communication pattern.

-   **Decoupling:** Publishers don't know who subscribers are; Subscribers don't know who publishers are.
-   **Asynchronous:** Sending a message doesn't block the sender (usually).
-   **Typed:** Data structure is strictly defined.

#### 2.2 Interface Definition Language (IDL)

ROS 2 uses `.msg` files to define interfaces. These are compiled into code for C++, Python, etc.

**Primitive Types:**
-   `bool`, `byte`, `char`
-   `float32`, `float64`
-   `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
-   `string`

**Complex Types:**
-   Arrays: `int32[]` (unbounded), `int32[5]` (fixed size).
-   Vectors: `int32[<=5]` (bounded vector).
-   Nesting: Using other message types inside a message.

**The Build Process (rosidl):**
1.  **Parse:** `.msg` file is parsed.
2.  **Generate IDL:** Converted to `.idl` (OMG IDL standard).
3.  **Generate Code:**
    -   C headers (`.h`)
    -   C++ headers (`.hpp`)
    -   Python modules (`.py`)
    -   **Type Support:** Code that tells the middleware how to serialize/deserialize this type.

#### 2.3 Serialization: CDR (Common Data Representation)

How does a C++ struct travel over the network? **Serialization**.

ROS 2 (via DDS) uses **CDR (Common Data Representation)**, a standard binary format defined by OMG.

**Process:**
1.  **User Code:** Populates a C++ object (e.g., `sensor_msgs::msg::Image`).
2.  **Type Support:** Converts C++ object -> CDR Serialized Stream.
    -   Handles endianness (Big Endian vs Little Endian).
    -   Handles alignment (padding bytes).
3.  **DDS:** Sends the CDR blob over UDP/SHM.
4.  **Receiver:** Deserializes CDR blob -> C++ object.

**Zero-Copy (Intra-process):**
If Publisher and Subscriber are in the same process, serialization is wasteful.
ROS 2 supports **Zero-Copy** via `rclcpp::NodeOptions().use_intra_process_comms(true)`.
-   Instead of serializing, a *pointer* to the message is passed.
-   Requires `std::unique_ptr` to ensure ownership safety (Publisher gives up ownership).

#### 2.4 The RMW Layer (ROS Middleware)

The **RMW** is the abstraction layer that sits between `rclcpp` and the specific DDS implementation (Fast DDS, Cyclone, etc.).

-   **Functions:** `rmw_create_node`, `rmw_publish`, `rmw_take`.
-   **Mapping:** Maps ROS concepts (Topics, Services) to DDS concepts (DataWriters, DataReaders).
-   **QoS Mapping:** Translates ROS QoS policies to DDS QoS policies.

**Why is this important for ADAS?**
Different RMW implementations have different performance characteristics.
-   **Cyclone DDS:** Generally better out-of-the-box for large payloads (LiDAR).
-   **Fast DDS:** Highly configurable, good for complex network topologies.
-   **Iceoryx:** True zero-copy shared memory transport (often used via RMW).

---

### 🔹 Part 3: Advanced Publisher/Subscriber Concepts

#### 3.1 Publisher Design Patterns

1.  **Periodic Publishing:**
    -   Driven by a `WallTimer`.
    -   Standard for sensors (e.g., 10Hz GPS).

2.  **Event-Driven Publishing:**
    -   Triggered by an external event (callback, interrupt).
    -   Example: Object detection result published only when image is received.

3.  **Lifecycle Publishing:**
    -   Using `rclcpp_lifecycle::LifecyclePublisher`.
    -   Only publishes when node is in `Active` state.
    -   *Mandatory for safety-critical nodes.*

#### 3.2 Subscriber Design Patterns

1.  **Direct Callback:**
    -   `create_subscription(topic, qos, callback)`
    -   Simple, but processes every message (subject to queue size).

2.  **Message Filters (Time Synchronizer):**
    -   Synchronizes messages from multiple topics based on timestamps.
    -   *Essential for Sensor Fusion (Camera + LiDAR).*

3.  **WaitSet (Advanced):**
    -   Manual waiting for messages.
    -   Allows complex logic (e.g., "wait for Camera OR LiDAR, but timeout after 10ms").
    -   Bypasses the Executor.

#### 3.3 QoS Depth & History

We covered QoS in Day 1, but let's look at **History** specifically for Topics.

-   **KEEP_LAST (Depth N):** Ring buffer. Oldest messages are overwritten.
    -   *ADAS:* Use `Depth=1` for latest sensor data (don't care about old images).
    -   *ADAS:* Use `Depth=10` for high-frequency TF transforms to allow interpolation.
-   **KEEP_ALL:** Stores everything until memory runs out.
    -   *ADAS:* Use for critical logs or "black box" recording.

---

## 💻 Implementation: Robust ADAS Node

We will implement a **Sensor Processing Node** that simulates reading data, processing it, and publishing results. We will use a custom message type.

### 🛠️ Package Setup

```bash
# Create package
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake \
  --dependencies rclcpp std_msgs sensor_msgs \
  --node-name sensor_node \
  adas_sensor_kit

# Create custom message directory
cd adas_sensor_kit
mkdir msg
```

### 📦 Defining Custom Messages

Create `msg/DetectedObject.msg`:

```text
# DetectedObject.msg
std_msgs/Header header

# Object ID and Class
uint32 object_id
string label
float32 confidence

# 3D Bounding Box (Center)
float32 position_x
float32 position_y
float32 position_z

# Dimensions
float32 width
float32 height
float32 depth

# Velocity
float32 velocity_x
float32 velocity_y
```

Create `msg/ObjectList.msg`:

```text
# ObjectList.msg
std_msgs/Header header
DetectedObject[] objects
```

**Update `CMakeLists.txt` for Message Generation:**

```cmake
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObject.msg"
  "msg/ObjectList.msg"
  DEPENDENCIES std_msgs
)
```

**Update `package.xml`:**

```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

### 👨‍💻 Node Implementation

We will create a node that:
1.  Subscribes to a (simulated) LiDAR point cloud.
2.  Subscribes to a (simulated) Camera image.
3.  Publishes the `ObjectList` custom message.
4.  Uses a `MultiThreadedExecutor` and `CallbackGroups`.

#### Header: `include/adas_sensor_kit/fusion_node.hpp`

```cpp
#ifndef ADAS_SENSOR_KIT__FUSION_NODE_HPP_
#define ADAS_SENSOR_KIT__FUSION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "adas_sensor_kit/msg/object_list.hpp"

#include <mutex>
#include <thread>
#include <vector>

namespace adas_sensor_kit
{

class FusionNode : public rclcpp::Node
{
public:
  explicit FusionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~FusionNode();

private:
  // Callbacks
  void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void camera_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void timer_callback();

  // Processing
  void process_fusion();

  // Subscriptions
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;

  // Publisher
  rclcpp::Publisher<adas_sensor_kit::msg::ObjectList>::SharedPtr objects_pub_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Callback Groups
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group_;
  rclcpp::CallbackGroup::SharedPtr camera_cb_group_;

  // Data Buffers (Thread Protected)
  std::mutex data_mutex_;
  sensor_msgs::msg::PointCloud2::SharedPtr last_lidar_;
  sensor_msgs::msg::Image::SharedPtr last_camera_;
};

}  // namespace adas_sensor_kit

#endif  // ADAS_SENSOR_KIT__FUSION_NODE_HPP_
```

#### Source: `src/fusion_node.cpp`

```cpp
#include "adas_sensor_kit/fusion_node.hpp"

using std::placeholders::_1;

namespace adas_sensor_kit
{

FusionNode::FusionNode(const rclcpp::NodeOptions & options)
: Node("fusion_node", options)
{
  // 1. Create Callback Groups
  // We use MutuallyExclusive groups for sensors to allow them to run in parallel
  // with each other, but not with themselves (no reentrancy for safety).
  lidar_cb_group_ = this->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  
  camera_cb_group_ = this->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);

  // 2. Configure Subscription Options
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = lidar_cb_group_;

  auto camera_sub_opt = rclcpp::SubscriptionOptions();
  camera_sub_opt.callback_group = camera_cb_group_;

  // 3. Create Subscriptions
  // QoS: SensorData (Best Effort, Volatile)
  lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "lidar_points", 
    rclcpp::SensorDataQoS(),
    std::bind(&FusionNode::lidar_callback, this, _1),
    lidar_sub_opt);

  camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "camera_image", 
    rclcpp::SensorDataQoS(),
    std::bind(&FusionNode::camera_callback, this, _1),
    camera_sub_opt);

  // 4. Create Publisher
  objects_pub_ = this->create_publisher<adas_sensor_kit::msg::ObjectList>(
    "detected_objects", 10);

  // 5. Create Timer (Processing Loop)
  // Runs at 20Hz
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(50),
    std::bind(&FusionNode::timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "Fusion Node Initialized with Multi-Threading Support");
}

FusionNode::~FusionNode()
{
  RCLCPP_INFO(this->get_logger(), "Fusion Node Shutting Down");
}

void FusionNode::lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // Simulate heavy processing
  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  
  std::lock_guard<std::mutex> lock(data_mutex_);
  last_lidar_ = msg;
  RCLCPP_DEBUG(this->get_logger(), "Received LiDAR frame");
}

void FusionNode::camera_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // Simulate heavy processing
  // std::this_thread::sleep_for(std::chrono::milliseconds(15));

  std::lock_guard<std::mutex> lock(data_mutex_);
  last_camera_ = msg;
  RCLCPP_DEBUG(this->get_logger(), "Received Camera frame");
}

void FusionNode::timer_callback()
{
  process_fusion();
}

void FusionNode::process_fusion()
{
  sensor_msgs::msg::PointCloud2::SharedPtr lidar_data;
  sensor_msgs::msg::Image::SharedPtr camera_data;

  // Critical Section: Copy data pointers locally to release lock quickly
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!last_lidar_ || !last_camera_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
        "Waiting for data...");
      return;
    }
    lidar_data = last_lidar_;
    camera_data = last_camera_;
  }

  // --- FUSION LOGIC (Simulated) ---
  
  auto msg = adas_sensor_kit::msg::ObjectList();
  msg.header.stamp = this->now();
  msg.header.frame_id = "base_link";

  // Create a dummy object
  adas_sensor_kit::msg::DetectedObject obj;
  obj.header.stamp = this->now();
  obj.header.frame_id = "base_link";
  obj.object_id = 1;
  obj.label = "Vehicle";
  obj.confidence = 0.95;
  obj.position_x = 10.5;
  obj.position_y = -2.3;
  obj.position_z = 0.0;
  obj.width = 1.8;
  obj.height = 1.5;
  obj.depth = 4.5;
  obj.velocity_x = 15.0; // m/s

  msg.objects.push_back(obj);

  // Publish
  objects_pub_->publish(msg);
  RCLCPP_INFO(this->get_logger(), "Published %zu objects", msg.objects.size());
}

}  // namespace adas_sensor_kit

// Main with MultiThreadedExecutor
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<adas_sensor_kit::FusionNode>();
  
  // Use MultiThreadedExecutor to allow parallel callback execution
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  
  executor.spin();
  
  rclcpp::shutdown();
  return 0;
}
```

### 🚀 Launch File

Create `launch/fusion_system.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='adas_sensor_kit',
            executable='fusion_node',
            name='fusion_node',
            output='screen',
            parameters=[
                {'use_sim_time': False}
            ]
        ),
        # We can add dummy publishers here to test
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
        )
    ])
```

---

## 🔬 Lab Exercise: Traffic Sign Message Design

### Lab Objectives
1.  Design a custom message for Traffic Sign Recognition.
2.  Implement a publisher that sends random traffic signs.
3.  Implement a subscriber that filters for "STOP" signs.

### Part 1: Message Design
Create `msg/TrafficSign.msg`:
```text
std_msgs/Header header
uint8 SIGN_TYPE_UNKNOWN=0
uint8 SIGN_TYPE_STOP=1
uint8 SIGN_TYPE_YIELD=2
uint8 SIGN_TYPE_SPEED_LIMIT=3

uint8 sign_type
float32 confidence
float32 distance
uint8 speed_limit_value # Only valid if type is SPEED_LIMIT
```

### Part 2: Publisher Logic
```cpp
// In a timer callback:
auto msg = adas_sensor_kit::msg::TrafficSign();
msg.sign_type = adas_sensor_kit::msg::TrafficSign::SIGN_TYPE_STOP;
msg.confidence = 0.99;
msg.distance = 15.0;
pub_->publish(msg);
```

### Part 3: Subscriber Logic
```cpp
void sign_callback(const adas_sensor_kit::msg::TrafficSign::SharedPtr msg) {
    if (msg->sign_type == adas_sensor_kit::msg::TrafficSign::SIGN_TYPE_STOP) {
        RCLCPP_WARN(this->get_logger(), "STOP SIGN DETECTED at %.2fm!", msg->distance);
        // Trigger braking logic...
    }
}
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Message not found" (Python/C++)
**Symptom:** `ImportError` or `#include` error for custom messages.
**Cause:** `CMakeLists.txt` or `package.xml` dependencies missing.
**Solution:**
-   Ensure `rosidl_generate_interfaces` is called.
-   Ensure `rosidl_default_generators` is in `package.xml` (build).
-   Ensure `rosidl_default_runtime` is in `package.xml` (exec).
-   **Source the workspace:** `source install/setup.bash` after build.

#### 2. Callback Starvation
**Symptom:** High-frequency topic works, but low-frequency timer never fires.
**Cause:** SingleThreadedExecutor is blocked by the high-freq callback.
**Solution:**
-   Use `MultiThreadedExecutor`.
-   Assign callbacks to different `CallbackGroups`.

#### 3. Topic Mismatch
**Symptom:** Nodes running, but no data received.
**Diagnostic:**
```bash
ros2 topic list -v
ros2 topic info /topic_name
ros2 doctor
```
**Check:**
-   Topic names match exactly (namespaces?).
-   Message types match exactly.
-   QoS profiles are compatible (Reliability/Durability).

### Introspection Tools

**1. `ros2 topic`**
```bash
ros2 topic list
ros2 topic echo /detected_objects
ros2 topic hz /lidar_points
ros2 topic bw /camera_image
```

**2. `rqt_graph`**
Visualizes the node topology.
```bash
rqt_graph
```

**3. `ros2 interface`**
Inspect message definitions.
```bash
ros2 interface show adas_sensor_kit/msg/DetectedObject
```

---

## ⚡ Optimization & Best Practices

### 1. Zero-Copy Transport
For large data (Images, PointClouds) between nodes in the **same process**:
-   Use `rclcpp::NodeOptions().use_intra_process_comms(true)`.
-   Publish `std::unique_ptr<Message>`.
-   Subscribe with `const Message::SharedPtr`.
-   *Result:* No serialization, no copy. Just pointer passing.

### 2. Const References
Always subscribe using `const SharedPtr &` or `const Msg &` to avoid unnecessary atomic reference counting operations if not needed.

```cpp
// Good
void callback(const sensor_msgs::msg::Image::SharedPtr msg)
// Better (if not storing the pointer)
void callback(const sensor_msgs::msg::Image & msg)
```

### 3. Executor Tuning
-   Don't spawn threads blindly.
-   Number of threads in `MultiThreadedExecutor` defaults to CPU core count.
-   For IO-bound tasks, this is fine. For CPU-bound tasks, be careful of context switching overhead.

### 4. Vectorization in Messages
-   Prefer `std::vector` (unbounded array in .msg) over many individual fields.
-   ROS 2 serialization of vectors is optimized (memcpy for POD types).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between a `SingleThreadedExecutor` and a `MultiThreadedExecutor`?
    *   **A:** Single executes callbacks sequentially in one thread (safe but blocking). Multi executes them in a thread pool (parallel but requires thread safety).

2.  **Q:** Why do we need `CallbackGroups`?
    *   **A:** To define which callbacks can run concurrently when using a MultiThreadedExecutor.

3.  **Q:** What is the purpose of the `.msg` file?
    *   **A:** To define a language-agnostic interface that generates C++/Python code and DDS IDL for serialization.

4.  **Q:** How does ROS 2 handle endianness during communication?
    *   **A:** CDR serialization handles it. The sender writes in its native endianness and flags it; the receiver swaps bytes if necessary.

### Challenge Task
**Task:** Implement a "Watchdog" Node.
1.  Subscribe to `/detected_objects`.
2.  If no message is received for 500ms, publish a `std_msgs/Bool` to `/emergency_stop` with `true`.
3.  Use a `WallTimer` to check the last received timestamp.

---

## 📚 Further Reading & References
-   [ROS 2 Executors Tutorial](https://docs.ros.org/en/humble/Concepts/About-Executors.html)
-   [Synchronization and Multithreading](https://docs.ros.org/en/humble/Concepts/About-Synchronization.html)
-   [DDS/ROS 2 QoS Policies](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)
-   [Fast DDS Configuration](https://fast-dds.docs.eprosima.com/)

---

**Day 2 Complete** | Phase 4: ADAS & Robotics Systems | Week 1: ROS 2 Fundamentals
