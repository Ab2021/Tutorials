# Day 146: Performance Optimization (Profiling/Tracing)
## Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone

---

> **📝 Day 146 Focus:**
> A self-driving car must react in milliseconds. If your Perception node takes 200ms, you might hit a pedestrian. **Profiling** helps us find the "slow" parts. We use **LTTng** and **ros2 trace** to visualize the flow of messages and measure latency.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Install** and **Run** `ros2 trace` (tracetools).
2.  **Visualize** trace data using **Trace Compass** or **Pandas**.
3.  **Identify** Callback Latency and Scheduling Jitter.
4.  **Optimize** node execution using `MultiThreadedExecutor`.
5.  **Implement** Zero-Copy communication (Loaned Messages).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **OS:** Linux Kernel Tracing (LTTng).
-   **ROS 2:** Executors and Callback Groups.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **ROS 2:** `ros2_tracing`, `tracetools_analysis`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Where does the time go?

Latency = $T_{pub} + T_{transport} + T_{queue} + T_{callback}$.
-   **Transport:** DDS serialization/deserialization.
-   **Queue:** Waiting in the subscriber queue.
-   **Callback:** Actual processing time (YOLO inference).

### 🔹 Part 2: Tracing vs Profiling

-   **Profiling (cProfile/Valgrind):** Tells you *which function* is slow (CPU usage).
-   **Tracing (LTTng):** Tells you *when* things happen (System events).
    -   "Message received at $t=1.000$".
    -   "Callback started at $t=1.005$".
    -   "Callback ended at $t=1.010$".

### 🔹 Part 3: Zero-Copy

Standard ROS 2 copies data: User Space $\to$ Kernel $\to$ Wire $\to$ Kernel $\to$ User Space.
**Zero-Copy (Loaned Messages):**
-   Publisher "borrows" memory from the middleware.
-   Subscriber "reads" directly from that memory (Shared Memory).
-   Requires compatible RMW (e.g., `rmw_cyclonedds_cpp` with Iceoryx).

---

## 💻 Implementation: Tracing a Node

**Scenario:**
-   A "Heavy" Node that sleeps for 50ms in callback.
-   We want to measure this latency.

### 🛠️ Setup
Create `week21_day146` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week21_day146
cd week21_day146/week21_day146
touch heavy_node.py
```

### 👨‍💻 Code: Heavy Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class HeavyNode(Node):
    def __init__(self):
        super().__init__('heavy_node')
        self.pub = self.create_publisher(String, 'topic', 10)
        self.sub = self.create_subscription(String, 'topic', self.callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback) # 10 Hz

    def timer_callback(self):
        msg = String()
        msg.data = "Hello"
        self.pub.publish(msg)

    def callback(self, msg):
        # Simulate heavy work
        time.sleep(0.05) # 50ms delay
        # self.get_logger().info('Processed')

def main(args=None):
    rclpy.init(args=args)
    node = HeavyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 🔬 Lab Exercise: Collecting a Trace

### Lab Objectives
1.  **Install Tracing Tools:**
    ```bash
    sudo apt install ros-humble-ros2-trace ros-humble-tracetools-analysis
    ```
2.  **Start Tracing:**
    ```bash
    ros2 trace --session-name my_trace
    ```
    *(This starts the LTTng session in the background).*
3.  **Run the Node:**
    In another terminal:
    ```bash
    ros2 run week21_day146 heavy_node
    ```
    Let it run for 10 seconds.
4.  **Stop Tracing:**
    Press `Enter` in the trace terminal.
    Output saved to `~/.ros/tracing/my_trace`.

### 📊 Analysis (Python)

Create `analyze_trace.py`.

```python
from tracetools_analysis.loading import load_file
from tracetools_analysis.processor.ros2 import Ros2Handler
from tracetools_analysis.utils.ros2 import Ros2DataModelUtil

# Load
path = '/home/user/.ros/tracing/my_trace' # Update user
events = load_file(path)

# Process
handler = Ros2Handler()
handler.process(events)
data = handler.data
util = Ros2DataModelUtil(data)

# Get Callback Durations
callbacks = util.get_callback_durations()

print("Callback Durations (ms):")
for cb in callbacks.iterrows():
    duration = cb[1]['duration'] * 1000 # s to ms
    print(f"{duration:.2f} ms")
```

**Expected Result:** You should see values around 50.0 ms.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "LTTng not found"
**Symptom:** `ros2 trace` fails.
**Cause:** Kernel modules not loaded or user not in `tracing` group.
**Solution:** `sudo usermod -aG tracing $USER` and reboot.

#### 2. Empty Trace
**Symptom:** Analysis shows no events.
**Cause:** Node didn't run long enough or instrumentation failed.
**Solution:** Ensure `LD_PRELOAD` is not interfering. Use `ros2 run --prefix 'ros2 trace --session-name ...'` to trace startup.

---

## ⚡ Optimization & Best Practices

### 1. Multi-Threaded Executor
By default, `rclpy.spin(node)` is Single-Threaded.
-   If you have 2 timers, they block each other.
-   **Solution:**
    ```python
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    ```
-   **Callback Groups:** Use `ReentrantCallbackGroup` to allow parallel execution of the same callback.

### 2. Loaned Messages (C++ Only)
Python doesn't support Zero-Copy well yet. In C++:
```cpp
auto msg = pub->borrow_loaned_message();
msg.get().data = ...;
pub->publish(std::move(msg));
```
This avoids `memcpy` entirely.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `SingleThreadedExecutor` and `MultiThreadedExecutor`?
    *   **A:** Single executes callbacks sequentially (FIFO). Multi executes them in parallel threads (if hardware allows).
2.  **Q:** What is "Jitter"?
    *   **A:** The variation in latency. If a 10Hz timer fires at 100ms, 105ms, 95ms, 110ms, the jitter is high. Real-time systems minimize jitter.
3.  **Q:** Why is Python bad for low-latency control?
    *   **A:** The Global Interpreter Lock (GIL) prevents true parallelism, and Garbage Collection causes unpredictable pauses.

### Challenge Task
**Task:** Executor Starvation.
1.  Add a second timer to `HeavyNode` (100Hz, fast).
2.  Run with `SingleThreadedExecutor`.
3.  Observe that the fast timer is blocked by the slow 50ms callback.
4.  Switch to `MultiThreadedExecutor` and `ReentrantCallbackGroup`.
5.  Observe that both run smoothly.

---

## 📚 Further Reading & References
-   [ROS 2 Tracing Documentation](https://github.com/ros2/ros2_tracing)
-   [Real-Time Programming in ROS 2](https://design.ros2.org/articles/realtime_background.html)

---

**Day 146 Complete** | Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone
