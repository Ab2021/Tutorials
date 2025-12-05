# Day 117: Traffic Management (OpenRMF)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 17: Swarm Robotics

---

> **📝 Content Creator Instructions:**
> Don't cross the streams.
> - **Focus:** Open Robotics Middleware Framework (OpenRMF), Traffic Conflict Deconfliction, Fleet Adapters, and Space-Time scheduling.
> - **Code:** A simplified Traffic Controller node that manages a "Intersection". Robots must request permission (Mutex) before entering the intersection zone.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the role of OpenRMF (Interoperability between heterogeneous fleets like Robots + Doors + Elevators).
2.  **Implement** a Simple Traffic Mutex (Critical Section) for shared zones.
3.  **Simulate** a Deadlock scenario (Two robots facing each other in a narrow hallway).
4.  **Interface** a dummy robot with a Fleet Adapter concept.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
# OpenRMF is strictly packaged. usage of Docker recommended.
# Here we simulate the concepts in pure ROS 2 python.
```

### Prior Knowledge
- Threading Locks (Mutex).
- Graph Navigation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Traffic Problem

*   **Scenario:** 10 Robots in a warehouse. 1 Narrow Aisle.
*   **Result:** Deadlock. Neither can move.
*   **Solution:** Centralized planning with negotiation.

### 🔹 Part 2: OpenRMF Concepts

*   **Fleet Adapter:** Converts "Robot-Specific" API to "RMF Standard" API.
*   **Traffic Schedule:** A database of "Who is where" at "What time".
    *   Robot A: Zone 1 from $t=0$ to $t=10$.
    *   Robot B: Zone 1 from $t=11$ to $t=20$.
*   **Negotiation:** If paths conflict, robots bid (cost function). Winner gets the path, Loser waits or reroutes.

### 🔹 Part 3: Critical Sections (Mutex)

Simplified version of Traffic Management:
*   Define a Zone (Polygon).
*   Robot asks Supervisor: "Can I enter Zone X?"
*   Supervisor checks: "Is Zone X empty?"
*   If Yes $\to$ Grant Token.
*   If No $\to$ Queue Request.

---

## 💻 Implementation: Intersection Controller

We will build a Centralized "Traffic Light" node for a shared intersection.

### 🛠️ Project Structure
```text
day117_traffic/
├── src/
│   ├── traffic_manager.py
│   └── robot_agent.py
└── launch/
    ├── intersection.launch.py
```

### 👨‍💻 Manager Node (`src/traffic_manager.py`)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

class TrafficManager(Node):
    def __init__(self):
        super().__init__('traffic_manager')
        
        # State: Who holds the lock?
        self.current_owner = None
        self.queue = []
        
        # Service-like logic via Topics (simplified)
        self.create_subscription(String, '/traffic/request', self.request_cb, 10)
        self.create_subscription(String, '/traffic/release', self.release_cb, 10)
        self.pub_status = self.create_publisher(String, '/traffic/status', 10)
        
        self.create_timer(0.5, self.broadcast_status)

    def request_cb(self, msg):
        robot_id = msg.data
        if self.current_owner is None:
            self.current_owner = robot_id
            self.get_logger().info(f"Granted lock to {robot_id}")
        else:
            if robot_id not in self.queue and robot_id != self.current_owner:
                self.queue.append(robot_id)
                self.get_logger().info(f"Queued {robot_id}")

    def release_cb(self, msg):
        robot_id = msg.data
        if self.current_owner == robot_id:
            self.get_logger().info(f"{robot_id} released lock")
            if len(self.queue) > 0:
                self.current_owner = self.queue.pop(0)
                self.get_logger().info(f"Granted lock to {self.current_owner}")
            else:
                self.current_owner = None

    def broadcast_status(self):
        msg = String()
        # Broadcast who has the lock. Robots check this.
        msg.data = str(self.current_owner) if self.current_owner else "FREE"
        self.pub_status.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(TrafficManager())
```

### 👨‍💻 Robot Agent (`src/robot_agent.py`)

Simulates a robot approaching an intersection.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class RobotAgent(Node):
    def __init__(self):
        super().__init__('robot_agent')
        self.id = self.get_namespace().strip('/')
        
        self.pub_req = self.create_publisher(String, '/traffic/request', 10)
        self.pub_rel = self.create_publisher(String, '/traffic/release', 10)
        self.sub_status = self.create_subscription(String, '/traffic/status', self.status_cb, 10)
        self.pub_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.has_lock = False
        self.state = "APPROACHING" 
        
        self.create_timer(0.1, self.loop)

    def status_cb(self, msg):
        owner = msg.data
        if owner == self.id:
            self.has_lock = True
        else:
            self.has_lock = False

    def loop(self):
        msg = Twist()
        
        if self.state == "APPROACHING":
            # Simulate approaching intersection
            msg.linear.x = 0.5
            # In real robot, check distance to zone
            # Here, just a timer simulation or manual transition
            if time.time() % 10 > 5: # Random trigger
                 self.state = "WAITING_FOR_LOCK"
                 req = String()
                 req.data = self.id
                 self.pub_req.publish(req)
                 self.get_logger().info("Requesting Lock...")
        
        elif self.state == "WAITING_FOR_LOCK":
            msg.linear.x = 0.0 # STOP
            if self.has_lock:
                self.state = "CROSSING"
                self.get_logger().info("Crossing Intersection...")
                self.cross_start_time = time.time()
                
        elif self.state == "CROSSING":
            msg.linear.x = 0.5
            if time.time() - self.cross_start_time > 3.0: # 3s to cross
                self.state = "DONE"
                rel = String()
                rel.data = self.id
                self.pub_rel.publish(rel)
                self.get_logger().info("Released Lock.")
                
        elif self.state == "DONE":
            msg.linear.x = 0.5
            
        self.pub_vel.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(RobotAgent())
```

---

## 🔬 Lab Exercise: "The 4-Way Stop"

### 1. Lab Objectives
- **Launch:** Traffic Manager + 4 Robots (North, South, East, West).
- **Goal:** All try to cross center.
- **Observe:**
    1.  R_North requests. Gets Lock. Crosses.
    2.  R_South, R_East, R_West queue up. They STOP.
    3.  R_North releases.
    4.  R_South gets Lock (FIFO). Crosses.
- **Fail Check:** Kill the Traffic Manager. Robots should STOP forever (Safety default) or switch to Local Avoidance.

---

## 🚀 Project: "Door Integration"

**Goal:** Robot opening an automatic door.
1.  **Node:** `door_adapter`.
2.  **Logic:**
    *   Robot approaches door.
    *   Sends `request_open` to Door Adapter.
    *   Door Adapter triggers `relay_on` (Hardware).
    *   Wait for `door_open_sensor` (Limit Switch).
    *   Adapter sends `door_state: OPEN`.
    *   Robot moves through.
    *   Robot sends `release_door`.
    *   Door closes.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Dangling Lock"
*   **Scenario:** Robot C gets lock, then crashes/runs out of battery inside the intersection.
*   **Result:** Queue waits forever.
*   **Fix:** Manager holds a "Time To Live" (TTL). If Robot C doesn't release in 10s, revoke lock and sound alarm.

#### 2. "Livelock"
*   **Scenario:** R1 waiting for R2. R2 waiting for R1.
*   **Fix:** Strict Ordering (Hierarchy) or Random Backoff.

---

## ⚡ Optimization: Predictive Scheduling

Instead of "Stop and Wait", tell the robot its slot in advance.
*   "Robot A, arrive at intersection at $t=15$. Robot B, arrive at $t=20$."
*   Robots adjust speed ($v = d/t$) to arrive exactly when the slot is open without stopping. (Energy Efficient).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** OpenRMF vs ROS 2 Navigation?
    *   **A:** Nav2 moves a *single* robot from A to B avoiding static obstacles. OpenRMF coordinates *multiple* robots to avoid dynamic congestion and share resources (lifts/doors).
2.  **Q:** What is a "Fleet Adapter"?
    *   **A:** The translation layer. It translates "RMF: Go to Waypoint X" into "vendor_specific_sdk.move_to(X)".
3.  **Q:** FIFO Queue?
    *   **A:** First In, First Out. Fairest simple strategy.

### Challenge Task
> **Task:** High Priority.
> 1. Add `ambulence_mode` to Robot 1.
> 2. If R1 requests lock, it jumps to front of queue.
> 3. Even if R2 has the lock, Manager tells R2 to "Abort/Move Aside".

---

## 📚 Further Reading
- **OpenRMF:** Official Documentation & Demos.
- **Multi-Agent Path Finding (MAPF):** Algorithms like CBS (Conflict-Based Search).

---

**Day 117 Complete**
