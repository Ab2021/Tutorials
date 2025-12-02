# Day 38: ROS 2 Architecture & DDS (Deep Dive)
## Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS

---

> **📝 Day 38 Focus:**
> ROS 1 was great for research. ROS 2 is built for production. The secret sauce? **DDS (Data Distribution Service)**. Today, we peel back the layers of ROS 2 to understand how it achieves real-time performance, reliability, and security without a central master.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** ROS 1 (TCPROS, Master) with ROS 2 (DDS, Distributed).
2.  **Explain** the DDS concepts: Domain ID, Participant, Topic, DataWriter, DataReader.
3.  **Configure** Quality of Service (QoS) policies: Reliability, Durability, History, Liveliness.
4.  **Switch** RMW implementations (FastDDS vs CycloneDDS) using environment variables.
5.  **Implement** a C++ ROS 2 node with custom QoS profiles for "Critical" vs "Sensor" data.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Networking:** UDP Multicast.
-   **C++:** Classes, Smart Pointers.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS with ROS 2 Humble/Jazzy.

### Software Stack
-   **ROS 2:** `rclcpp`.
-   **DDS Tools:** `ros2 doctor`, `ros2 topic`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why ROS 2?

**ROS 1 Issues:**
-   **Single Point of Failure:** The `roscore` (Master). If it dies, the robot dies.
-   **Not Real-Time:** TCP is reliable but non-deterministic (retries, buffering).
-   **Security:** None. Any node can publish to any topic.

**ROS 2 Solutions:**
-   **Distributed:** No Master. Nodes discover each other via UDP Multicast.
-   **Real-Time Ready:** Uses DDS, which supports real-time QoS.
-   **Secure:** SROS2 supports encryption and authentication.

### 🔹 Part 2: The DDS Standard

**DDS (Data Distribution Service)** is an industry standard (OMG) used in battleships, dams, and finance. ROS 2 sits *on top* of DDS.

#### 2.1 Architecture
-   **Global Data Space:** All nodes share a virtual data space.
-   **Domain ID:** Virtual partition (0-232). Nodes on Domain 0 cannot see Domain 1.
-   **RTPS (Real-Time Publish Subscribe):** The wire protocol (UDP-based).

#### 2.2 RMW (ROS Middleware)
ROS 2 abstracts DDS via the RMW layer. You can swap the backend:
-   `rmw_fastrtps_cpp` (eProsima FastDDS) - Default.
-   `rmw_cyclonedds_cpp` (Eclipse CycloneDDS) - Lightweight, good for WiFi.
-   `rmw_connext_cpp` (RTI Connext) - Commercial, certified for safety.

### 🔹 Part 3: Quality of Service (QoS)

In TCP, you get "Reliable". In UDP, you get "Best Effort".
DDS gives you fine-grained control.

1.  **Reliability:**
    -   **Reliable:** Guarantees delivery (Retries). Like TCP. Used for Services/Parameters.
    -   **Best Effort:** Fire and forget. Like UDP. Used for Sensor Data (Lidar/Camera).
2.  **Durability:**
    -   **Volatile:** Late joiners miss past messages.
    -   **Transient Local:** Late joiners get the *last* message (Latch). Used for Maps/TF Static.
3.  **History:**
    -   **Keep Last (N):** Keep only N messages.
    -   **Keep All:** Keep everything (until RAM runs out).
4.  **Deadline:**
    -   "I expect a message every 100ms." If not, trigger an event.

**Compatibility Rule:** Publisher QoS must be *at least as good* as Subscriber QoS.
-   Pub(Best Effort) -> Sub(Reliable) = **Incompatible** (No connection).
-   Pub(Reliable) -> Sub(Best Effort) = **Compatible**.

---

## 💻 Implementation: Custom QoS in C++

We will write a ROS 2 package `qos_demo` with two nodes:
1.  `sensor_pub`: Publishes Lidar data (Best Effort).
2.  `critical_pub`: Publishes Emergency Stop (Reliable + Transient Local).

### 🛠️ Setup
Create `week6_day38` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake week6_day38
cd week6_day38
touch src/qos_demo.cpp
```

### 👨‍💻 Code: qos_demo.cpp

```cpp
#include <chrono>
#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class QoSNode : public rclcpp::Node {
public:
    QoSNode() : Node("qos_demo_node") {
        // --- 1. Sensor Profile (Best Effort) ---
        // We don't care if we miss a packet, we want the latest data fast.
        rclcpp::QoS sensor_qos(10); // Keep Last 10
        sensor_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        sensor_qos.durability(rclcpp::DurabilityPolicy::Volatile);

        sensor_pub_ = this->create_publisher<std_msgs::msg::String>("lidar_scan", sensor_qos);
        
        // --- 2. Critical Profile (Reliable + Latch) ---
        // We MUST ensure the subscriber gets this, even if they join late.
        rclcpp::QoS critical_qos(1); // Keep Last 1
        critical_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        critical_qos.durability(rclcpp::DurabilityPolicy::TransientLocal); // Latch

        critical_pub_ = this->create_publisher<std_msgs::msg::String>("emergency_stop", critical_qos);

        timer_ = this->create_wall_timer(500ms, std::bind(&QoSNode::timer_callback, this));
    }

private:
    void timer_callback() {
        auto msg = std_msgs::msg::String();
        
        // Publish Sensor Data
        msg.data = "Scan Data " + std::to_string(count_);
        sensor_pub_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Published Sensor: %s", msg.data.c_str());

        // Publish Critical Data (Only once every 10 counts for demo)
        if (count_ % 10 == 0) {
            msg.data = "STOP NOW!";
            critical_pub_->publish(msg);
            RCLCPP_INFO(this->get_logger(), "Published Critical: %s", msg.data.c_str());
        }
        
        count_++;
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr sensor_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr critical_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    size_t count_ = 0;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<QoSNode>());
    rclcpp::shutdown();
    return 0;
}
```

### 👨‍💻 Code: CMakeLists.txt

Add this to `CMakeLists.txt`:

```cmake
add_executable(qos_demo src/qos_demo.cpp)
ament_target_dependencies(qos_demo rclcpp std_msgs)

install(TARGETS
  qos_demo
  DESTINATION lib/${PROJECT_NAME})
```

### 🛠️ Build & Run

```bash
cd ~/ros2_ws
colcon build --packages-select week6_day38
source install/setup.bash
ros2 run week6_day38 qos_demo
```

---

## 🔬 Lab Exercise: QoS Incompatibility

### Lab Objectives
1.  Run the `qos_demo` node.
2.  **Experiment A:** Try to subscribe to `lidar_scan` with default QoS (Reliable).
    ```bash
    ros2 topic echo /lidar_scan --qos-reliability reliable
    ```
    -   *Result:* **Silence.** The publisher is Best Effort, but you asked for Reliable. Incompatible.
3.  **Experiment B:** Subscribe with Best Effort.
    ```bash
    ros2 topic echo /lidar_scan --qos-reliability best_effort
    ```
    -   *Result:* Data appears.
4.  **Experiment C:** Late Joining.
    -   Wait for "STOP NOW!" to be published.
    -   Start `ros2 topic echo /emergency_stop --qos-durability transient_local`.
    -   *Result:* You immediately receive the last "STOP NOW!" message, even though it was sent in the past. This is **Transient Local** (Latching).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Nodes don't see each other
**Symptom:** `ros2 node list` is empty or missing nodes.
**Cause:**
-   Different `ROS_DOMAIN_ID`. Check `echo $ROS_DOMAIN_ID`.
-   Firewall blocking UDP Multicast.
-   Multicast disabled on loopback (if running locally).
**Solution:** `export ROS_DOMAIN_ID=0`. Enable multicast: `sudo ifconfig lo multicast`.

#### 2. High CPU Usage (FastDDS)
**Symptom:** `ros2_daemon` eating CPU.
**Cause:** Discovery traffic flooding the network.
**Solution:** Use a **Discovery Server** instead of Multicast (for large networks). Or switch to CycloneDDS.

#### 3. Large Data Dropped
**Symptom:** Camera images flickering.
**Cause:** UDP packet size limit (MTU ~1500 bytes). Fragmentation issues.
**Solution:** Increase OS UDP buffer sizes (`sysctl -w net.core.rmem_max=26214400`).

---

## ⚡ Optimization & Best Practices

### 1. Zero Copy (Intra-process)
If two nodes are in the same process (Components), passing messages via DDS involves serialization (slow).
**Intra-process Communication:**
-   ROS 2 can bypass DDS and pass a pointer directly.
-   Enable it: `options.use_intra_process_comms(true)`.
-   Use `unique_ptr` messages to transfer ownership.

### 2. Choosing RMW
-   **WiFi:** Use CycloneDDS (Better handling of lossy networks).
-   **Ethernet/Local:** Use FastDDS (High throughput).
-   **Switching:** `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the default Reliability QoS in ROS 2?
    *   **A:** Reliable (for Services/Parameters) and Reliable (for Topics, usually). But `SensorData` profile defaults to Best Effort.
2.  **Q:** How does a node find another node without a Master?
    *   **A:** UDP Multicast (Simple Discovery Protocol). "I am here, I publish X." "I am here, I subscribe to X."
3.  **Q:** What happens if I set `ROS_DOMAIN_ID=10` on one robot and `20` on another?
    *   **A:** They will be completely isolated and cannot communicate.

### Challenge Task
**Task:** Discovery Server.
1.  Configure FastDDS to use a Discovery Server (Centralized discovery, but decentralized data).
2.  Run the server: `fastdds discovery -i 0`.
3.  Point nodes to it: `export ROS_DISCOVERY_SERVER=127.0.0.1:11811`.
4.  Observe reduced network traffic (Wireshark).

---

## 📚 Further Reading & References
-   [ROS 2 QoS Design Guide](https://design.ros2.org/articles/qos.html)
-   [DDS Specification (OMG)](https://www.omg.org/spec/DDS/)

---

**Day 38 Complete** | Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS
