# Day 100: Quality of Service (QoS) & DDS Tuning
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 15: Production-Grade ROS 2

---

> **📝 Content Creator Instructions:**
> UDP vs TCP was the old way. QoS is the ROS 2 way.
> - **Focus:** DDS (Data Distribution Service) concepts: Reliability, Durability, History, and Liveliness.
> - **Code:** A demonstration of "Transient Local" durability where a Subscriber receives a message sent *before* it even started (Latched Topic behavior equivalent).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** QoS profiles for different use cases: Sensor Data (Best Effort) vs Parameters (Reliable).
2.  **Explain** "Transient Local" Durability (The "Late Joiner" problem).
3.  **Tune** DDS XML configurations (`cyclonedds.xml` / `fastdds.xml`) for high-bandwidth Wi-Fi.
4.  **Debug** QoS Mismatches (Why isn't my subscriber receiving data?).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### Prior Knowledge
- TCP/UDP Protocols.
- ROS 2 Topics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The QoS Policies

1.  **Reliability:**
    *   **Best Effort:** UDP-like. Fire and forget. Fast. Good for Lidar/Video (if you miss a frame, just wait for next).
    *   **Reliable:** TCP-like. Retries until ack. Good for Services/Parameters/Maps.
2.  **Durability:**
    *   **Volatile:** Messages lost if no one is listening *now*.
    *   **Transient Local:** Publisher stores `N` messages. New subscribers get them immediately upon connection. (Old ROS 1 "Latched" topic).
3.  **History:**
    *   **Keep Last (N):** Circular buffer.
    *   **Keep All:** Store everything (Dangerous on memory).

### 🔹 Part 2: The Compatibility Rule

**Publisher and Subscriber must be compatible.**
*   Reliability: Pub `Best Effort` $\to$ Sub `Reliable` = **Incompatible** (Sub expects guarantees Pub won't give).
*   Reliability: Pub `Reliable` $\to$ Sub `Best Effort` = **Compatible**.
*   Durability: Pub `Volatile` $\to$ Sub `Transient Local` = **Compatible** (but won't get past data).

### 🔹 Part 3: DDS Implementation Tuning

DDS Middleware does the heavy lifting.
*   **Discovery:** Multicast (Default) vs Unicast (Cloud/VPN).
*   **Fragment Size:** Increasing UDP packet size can boost throughput for PointClouds.

---

## 💻 Implementation: The Late Joiner

We will create a "Map Server" (Transient Local) and a "Navigator" (Late subscriber).

### 🛠️ Project Structure
```text
day100_qos/
├── src/
│   ├── map_publisher.cpp
│   └── late_subscriber.cpp
└── config/
    └── qos_profiles.yaml
```

### 👨‍💻 Map Publisher (`src/map_publisher.cpp`)

Sends the map ONCE, then shuts up.

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class MapPublisher : public rclcpp::Node
{
public:
  MapPublisher() : Node("map_pub")
  {
    // Transient Local QoS
    rclcpp::QoS qos_profile(1); // Keep Last 1
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

    pub_ = this->create_publisher<std_msgs::msg::String>("static_map", qos_profile);
    
    // Publish once after 1 second
    timer_ = this->create_wall_timer(std::chrono::seconds(1), [this](){
        auto msg = std_msgs::msg::String();
        msg.data = "HUGE_MAP_DATA_JSON_BLOB";
        RCLCPP_INFO(this->get_logger(), "Publishing Map...");
        pub_->publish(msg);
        timer_->cancel(); // Stop
    });
  }

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MapPublisher>());
  rclcpp::shutdown();
  return 0;
}
```

### 👨‍💻 Late Subscriber (`src/late_subscriber.cpp`)

Starts 10 seconds later.

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class LateSubscriber : public rclcpp::Node
{
public:
  LateSubscriber() : Node("late_sub")
  {
    // Must match Pub's Durability!
    rclcpp::QoS qos_profile(1);
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    
    // If we used default (Volatile), we would NEVER receive the map 
    // because it was sent in the past.

    sub_ = this->create_subscription<std_msgs::msg::String>(
      "static_map", 
      qos_profile,
      [this](std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received Map: %s", msg->data.c_str());
      });
  }

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LateSubscriber>());
  rclcpp::shutdown();
  return 0;
}
```

---

## 🔬 Lab Exercise: "The Mismatch"

### 1. Lab Objectives
- **Run MapPub:** Sends message at $t=1$.
- **Wait:** $t=5$.
- **Run LateSub (Correct QoS):** Receives message immediately.
- **Modify:** Change LateSub QoS to `Volatile` (Default).
- **Run:** Receives NOTHING.
- **Diagnosis:** Use `ros2 topic info /static_map --verbose`.
    *   See "Offered QoS" vs "Requested QoS".

---

## 🚀 Project: "DDS Tuning for Wi-Fi"

**Goal:** Stream 4K Video over Wi-Fi.
1.  **Default:** Laggy, dropped frames.
2.  **Config:** Create `cyclonedds.xml`.
    *   Increase `MaxMessageSize`.
    *   Increase `FragmentSize`.
    *   Set `HistoryDepth` to small (don't buffer old video).
3.  **Apply:** `export CYCLONEDDS_URI=file://cyclonedds.xml`.
4.  **Result:** Smooth streaming.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Incompatible QoS"
*   **Symptom:** Publisher running, Subscriber running, Topic list shows both. NO DATA.
*   **Fix:** `ros2 topic info /topic -v`. Look for `Reliability: best effort` on Pub and `Reliability: reliable` on Sub.

#### 2. "Discovery storm"
*   **Symptom:** CPU 100% on network startup.
*   **Cause:** Multicast flooding if too many nodes (100+).
*   **Fix:** Use `ROS_DISCOVERY_SERVER` or configure DDS to use Unicast list.

---

## ⚡ Optimization: Zero-Copy (Loaned Messages)

QoS isn't just about delivery. It's about memory.
*   **Loaned Message:** `pub->borrow_loaned_message()`.
*   Asks middleware for memory chunk.
*   User writes directly to chunk.
*   Middleware sends chunk.
*   **No serialization copy.** (Requires `rmw_cyclonedds_cpp` or `fastrtps` with Shared Memory enabled).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Which QoS for PointClouds?
    *   **A:** Best Effort, Volatile, Keep Last 1. We don't care about old scans, and we don't want to retry dropped packets (latency).
2.  **Q:** Which QoS for Robot Description (URDF)?
    *   **A:** Reliable, Transient Local. It's sent once, and every new node needs to see it.
3.  **Q:** What is RMW?
    *   **A:** ROS Middleware. The interface between ROS 2 client lib and the underlying DDS (Data Distribution Service).

### Challenge Task
> **Task:** Deadline QoS.
> 1. Set `Deadline` to 100ms.
> 2. Publish at 5Hz (200ms).
> 3. Subscriber (or Event Handler) should report "Deadline Missed".
> 4. Useful for Safety Watchdogs.

---

## 📚 Further Reading
- **ROS 2 Docs:** "About QoS Settings".
- **CycloneDDS:** "Configuration Guide".

---

**Day 100 Complete**
