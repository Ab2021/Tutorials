# Day 113: Swarm Architecture
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 17: Swarm Robotics

---

> **📝 Content Creator Instructions:**
> One robot is a toy. Ten robots are a team. A hundred robots are a swarm.
> - **Focus:** Centralized vs Decentralized architectures, Scalability, Robustness, and Stigmergy.
> - **Code:** A "Ping" node that simulates neighbor discovery. Each robot broadcasts "I am here" and builds a local list of neighbors within communication range.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Compare** Centralized (Cloud-based) vs Decentralized (Mesh-based) control.
2.  **Implement** a Neighbor Discovery protocol using ROS 2 Topics (`/robot_N/heartbeat`).
3.  **Simulate** Bandwidth limitations (Scalability wall).
4.  **Architect** a system where "The Swarm survives even if 50% of robots fail".

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Multiple robots (or multiple namespaces in simulation).

### Software Environment
```bash
# Standard ROS 2
```

### Prior Knowledge
- ROS 2 Namespaces (`/robot_1/`, `/robot_2/`).
- Discovery (DDS).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Scalability Problem

*   **Centralized:** 1 Server controls 100 Robots.
    *   *Pros:* Global Optimization (Perfect Path Planning).
    *   *Cons:* Single Point of Failure. Bandwidth bottleneck. Latency.
*   **Decentralized:** 100 Robots talk to neighbors.
    *   *Pros:* Robust. Infinite Scalability (locally).
    *   *Cons:* Local Minima (Traffic Jams). No global truth.

### 🔹 Part 2: Swarm Intelligence

Biologically inspired (Ants, Bees, Birds).
*   **Stigmergy:** Indirect coordination via the environment (Ants leaving pheromones).
*   **Emergent Behavior:** Complex global patterns arise from simple local rules (e.g., Flocking).

### 🔹 Part 3: Architecture in ROS 2

*   **Namespaces:** Essential. `ros2 launch robot_ns.launch.py id:=1`.
*   **topics:**
    *   Global: `/swarm/task` (Broadcast).
    *   Local: `/robot_1/odom`, `/robot_2/scan`.
*   **Communication:**
    *   **All-to-All:** $O(N^2)$. Bad.
    *   **Local-Unknown:** Sub to `/robot_*/odom`. Good.

---

## 💻 Implementation: Neighbor Discovery

We will write a node that listens for heartbeats and tracks who is alive.

### 🛠️ Project Structure
```text
day113_swarm/
├── src/
│   ├── heartbeat_node.py
│   └── neighbor_monitor.py
└── launch/
    └── swarm_sim.launch.py
```

### 👨‍💻 Heartbeat Node (`src/heartbeat_node.py`)

Runs on every robot.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HeartbeatNode(Node):
    def __init__(self):
        super().__init__('heartbeat')
        
        # Get Namespace ID
        self.robot_id = self.get_namespace().strip('/')
        
        # Publisher
        self.pub = self.create_publisher(String, '/swarm/heartbeats', 10)
        
        # Timer (1Hz)
        self.create_timer(1.0, self.pub_heartbeat)

    def pub_heartbeat(self):
        msg = String()
        # Payload: "ID, X, Y, Status" ( Simplified)
        msg.data = f"{self.robot_id},ACTIVE"
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(HeartbeatNode())
```

### 👨‍💻 Monitor Node (`src/neighbor_monitor.py`)

Each robot runs this too, or just a central monitor.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class NeighborMonitor(Node):
    def __init__(self):
        super().__init__('neighbor_monitor')
        self.sub = self.create_subscription(String, '/swarm/heartbeats', self.cb, 100)
        
        # Database: {id: last_seen_time}
        self.neighbors = {}
        
        self.create_timer(1.0, self.cleanup)

    def cb(self, msg):
        rid, status = msg.data.split(',')
        self.neighbors[rid] = time.time()

    def cleanup(self):
        now = time.time()
        # Remove dead robots (> 3s silence)
        alive = [rid for rid, last in self.neighbors.items() if now - last < 3.0]
        self.get_logger().info(f"Swarm Count: {len(alive)} | Members: {alive}")

def main():
    rclpy.init()
    rclpy.spin(NeighborMonitor())
```

### 👨‍💻 Launch (`launch/swarm_sim.launch.py`)

Spawns 5 robots.

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace

def generate_launch_description():
    ld = LaunchDescription()
    
    for i in range(5):
        robot_name = f'robot_{i}'
        
        node = Node(
            package='day113_swarm',
            executable='heartbeat',
            namespace=robot_name
        )
        
        ld.add_action(node)
        
    monitor = Node(
        package='day113_swarm',
        executable='neighbor_monitor',
        name='monitor'
    )
    ld.add_action(monitor)
    
    return ld
```

---

## 🔬 Lab Exercise: "Kill Logic"

### 1. Lab Objectives
- **Launch:** `ros2 launch day113_swarm swarm_sim.launch.py`.
- **Observe:** Monitor reports "Swarm Count: 5".
- **Kill:** Open specific terminal (or kill process) for `robot_3`.
- **Observe:** Monitor reports "Swarm Count: 4".
- **Restore:** Restart `robot_3`.
- **Observe:** Count goes back to 5.
- **Concept:** Self-Healing Discovery.

---

## 🚀 Project: "The Bucket Brigade"

**Goal:** Pass a message from Robot 1 to Robot 5 via neighbors.
1.  **Rule:** Robot N can only talk to Robot N-1 and N+1.
2.  **Scenario:** Robot 1 detects "Fire".
3.  **Process:**
    *   R1 pub `/r1/alert`.
    *   R2 sub `/r1/alert`, then pub `/r2/alert`.
    *   ...
    *   R5 sub `/r4/alert`, then Calls 911.
4.  **Failure:** Kill R3. Does the message stop? (Yes, in a chain. No, in a Mesh).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Domain ID Conflict"
*   **Cause:** Creating 100 nodes on DDS might hit OS limits (files handles/ports).
*   **Fix:** Use `CycloneDDS` with optimized config or `Zenoh` router for large swarms. ROS 2 default limits are often around 100-200 nodes per machine.

#### 2. "Namespace Hell"
*   **Symptom:** `/cmd_vel` instead of `/robot_1/cmd_vel`.
*   **Fix:** Always use `PushRosNamespace` in Launch files and relative topic names (`cmd_vel`, not `/cmd_vel`) in C++/Python.

---

## ⚡ Optimization: Spatial Partitioning

If you have 1000 robots, `/swarm/heartbeats` floods the network ($1000 \times 1000 = 1M$ messages).
*   **Fix:** Spatial Partitioning.
*   Robot only subscribes to topics relevant to its Grid Cell.
*   Or use **Range-Limited comms** simulation where packets drop if distance > $R$.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the main advantage of Decentralized control?
    *   **A:** Robustness. No single point of failure. If the leader dies, the swarm adapts.
2.  **Q:** How do we handle unique parameters for 100 robots?
    *   **A:** Param files with wildcards or namespaced yaml: `robot_1: ros__parameters: ...`.
3.  **Q:** What is "Boids"?
    *   **A:** Flocking algorithm. Separation (Don't crash), Alignment (Steer same way), Cohesion (Stay close).

### Challenge Task
> **Task:** Leader Election.
> 1. Start 5 robots with random IDs.
> 2. They broadcast IDs.
> 3. The one with Highest ID becomes LEADER.
> 4. If Leader dies, Highest Remaining becomes LEADER.

---

## 📚 Further Reading
- **Swarm Robotics:** "The Boids Algorithm" (Reynolds).
- **ROS 2:** "Multi-Robot spawning in Gazebo".

---

**Day 113 Complete**
