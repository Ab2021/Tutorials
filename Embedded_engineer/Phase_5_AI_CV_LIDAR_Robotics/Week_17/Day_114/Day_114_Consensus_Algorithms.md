# Day 114: Consensus Algorithms (Leader Election)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 17: Swarm Robotics

---

> **📝 Content Creator Instructions:**
> Who is in charge? One of us.
> - **Focus:** Distributed Consensus (Raft/Paxos/Bully Algorithm), Handling Split-Brain (Network Partitions), and Synchronization.
> - **Code:** A Python implementation of the **Bully Algorithm**. Robots elect a leader based on the highest unique ID. If the leader dies, they re-elect.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why Consensus is hard (The Two Generals Problem).
2.  **Implement** the Bully Algorithm for Leader Election in ROS 2.
3.  **Handle** Network Partitions (Split Brain scenarios).
4.  **Synchronize** specific actions (e.g., "Attack at dawn" / "Lift box together").

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Swarm of 3+ simulated robots.

### Software Environment
```bash
# Standard ROS 2
```

### Prior Knowledge
- Neighbor Discovery (Day 113).
- State Machines.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Consensus Problem

Robots need to agree on a shared state (e.g., "Current Mission", "Who is Leader") without a central server.
*   **FLP Impossibility:** In a fully asynchronous system, consensus is impossible if one node can fail. (Theoretical limit).
*   **Practical Solution:** Timeouts. If Leader doesn't reply in $T$ seconds, assume dead.

### 🔹 Part 2: The Bully Algorithm

A simple, robust algorithm for small swarms.
1.  **Selection:** The node with the **Highest ID** is the Leader.
2.  **Election:**
    *   If P detects Leader is dead, it sends `ELECTION` to all nodes with $ID > P$.
    *   If no one answers, P declares itself Leader.
    *   If Q ($ID > P$) answers, Q takes over the election.
3.  **Victory:** The winner sends `COORDINATOR` message to everyone.

### 🔹 Part 3: Split-Brain

What if the WiFi breaks in half?
*   Group A (Robots 1, 2) sees Robot 2 as leader.
*   Group B (Robots 3, 4, 5) sees Robot 5 as leader.
*   **Result:** Two leaders. Conflict when network heals.
*   **Fix:** Quorum (Majority). You need $> N/2$ votes to be leader. Group A (2 robots) < 3, so they go passive.

---

## 💻 Implementation: The Bully Node

Each robot runs this logic.

### 🛠️ Project Structure
```text
day114_consensus/
├── src/
│   ├── bully_node.py
└── launch/
    └── election.launch.py
```

### 👨‍💻 Bully Node (`src/bully_node.py`)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
import time

class BullyNode(Node):
    def __init__(self):
        super().__init__('bully_node')
        
        # Get ID from param or namespace
        ns = self.get_namespace().strip('/')
        self.my_id = int(ns.split('_')[-1]) # robot_3 -> 3
        
        # State
        self.leader_id = -1
        self.leader_last_seen = 0
        self.election_in_progress = False
        
        # Topics
        self.pub_election = self.create_publisher(Int32, '/swarm/election', 10) # "I want to be leader"
        self.pub_coordinator = self.create_publisher(Int32, '/swarm/coordinator', 10) # "I AM leader"
        self.pub_heartbeat = self.create_publisher(Int32, '/swarm/heartbeat', 10)
        
        self.create_subscription(Int32, '/swarm/election', self.election_cb, 10)
        self.create_subscription(Int32, '/swarm/coordinator', self.coord_cb, 10)
        self.create_subscription(Int32, '/swarm/heartbeat', self.heartbeat_cb, 10)
        
        # Timer (Check leader health)
        self.create_timer(1.0, self.check_leader)
        # Timer (Send heartbeat if I am leader)
        self.create_timer(0.5, self.send_heartbeat)

    def heartbeat_cb(self, msg):
        sender_id = msg.data
        if sender_id == self.leader_id:
            self.leader_last_seen = time.time()
        
        # If I hear a heartbeat from a HIGHER ID than me, I submit.
        if sender_id > self.my_id and self.leader_id == self.my_id:
            self.get_logger().info(f"Stepping down. {sender_id} is bigger.")
            self.leader_id = sender_id

    def election_cb(self, msg):
        sender_id = msg.data
        if sender_id < self.my_id:
            # Someone smaller started election. I bully them.
            # "Silence, peasant! I will hold the election."
            msg = Int32()
            msg.data = self.my_id
            self.pub_election.publish(msg) # Restart election with my ID
            self.start_election()

    def coord_cb(self, msg):
        # Someone won
        self.leader_id = msg.data
        self.leader_last_seen = time.time()
        self.election_in_progress = False
        self.get_logger().info(f"New Leader: robot_{self.leader_id}")

    def check_leader(self):
        if self.leader_id == self.my_id:
            return # I am king.

        # If Leader dead or undefined
        if time.time() - self.leader_last_seen > 3.0:
            if not self.election_in_progress:
                self.get_logger().warn("Leader dead! Starting election.")
                self.start_election()

    def start_election(self):
        self.election_in_progress = True
        # Announce candidacy
        msg = Int32()
        msg.data = self.my_id
        self.pub_election.publish(msg)
        
        # Wait a bit. If no one with higher ID speaks, I win.
        # In real Bully, we wait for "OK" messages. 
        # Simplified: If no *Higher* Election msg seen in 1s, I win.
        self.create_timer(1.0, self.declare_victory) 
        # Note: Need logic to cancel this timer if I hear higher ID.

    def declare_victory(self):
        if self.election_in_progress:
            self.get_logger().info("I AM THE CAPTAIN NOW.")
            self.leader_id = self.my_id
            msg = Int32()
            msg.data = self.my_id
            self.pub_coordinator.publish(msg)
            self.election_in_progress = False

    def send_heartbeat(self):
        if self.leader_id == self.my_id:
            msg = Int32()
            msg.data = self.my_id
            self.pub_heartbeat.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(BullyNode())
```

---

## 🔬 Lab Exercise: "The Coup"

### 1. Lab Objectives
- **Launch:** 5 Robots (IDs 0, 1, 2, 3, 4).
- **Observe:** Robot 4 declares victory.
- **Kill:** Kill Robot 4.
- **Observe:** Heartbeats stop.
- **Action:** Robot 3 detects timeout $\to$ Starts Election $\to$ Robot 0,1,2 see R3 is bigger $\to$ Robot 3 Wins.
- **Revive:** Start Robot 4 again.
- **Observe:** R4 sends Heartbeat. R3 sees $ID_4 > ID_3$. R3 steps down. R4 resumes leadership.

---

## 🚀 Project: "Distributed Clock Sync"

**Goal:** Execute action exactly at $t=100.0s$.
1.  **Problem:** Robot clocks drift.
2.  **Algorithm:** Berkeley Algorithm.
3.  **Process:**
    *   Leader polls all followers for their time.
    *   Leader calculates average difference.
    *   Leader tells each follower: "Adjust by $+ \delta t$".
4.  **Result:** Synchronized dance moves.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Election loop"
*   **Scneario:** R1 and R2 keep starting elections instantly.
*   **Cause:** Timeout is too short compared to network latency.
*   **Fix:** Add random jitter to election timeout (Raft approach).

#### 2. "Ghost Leader"
*   **Scenario:** R4 crashes but socket keeps port open? (Unlikely in ROS).
*   **Scenario:** Intermittent WiFi makes R4 appear/disappear.
*   **Result:** Constant re-elections (System Paralysis).
*   **Fix:** Hysteresis. Require 3 missed heartbeats to kill. Require 3 successful heartbeats to accept new leader.

---

## ⚡ Optimization: Token Ring

Instead of bullying, pass a "Token".
*   R1 $\to$ R2 $\to$ R3 $\to$ R1.
*   Only the token holder can speak/lead.
*   **Pros:** Fair. Deterministic bandwidth.
*   **Cons:** If token is lost (robot dies holding it), you need complex regeneration logic.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not use Raft?
    *   **A:** Raft is complex (Log replication, Safety). Bully is simpler for just "Who is the boss" without shared storage state.
2.  **Q:** What is Split Brain?
    *   **A:** When the network partitions, and both halves elect a leader. When merged, you have two leaders fighting.
3.  **Q:** Higher ID wins?
    *   **A:** Arbitrary rule. Could be "Highest Battery Level" wins.

### Challenge Task
> **Task:** Battery-Based Leadership.
> 1. Modify `BullyNode`.
> 2. Instead of `my_id`, use `battery_level`.
> 3. Verify that as Leader's battery drains, a fresher robot takes over automatically.

---

## 📚 Further Reading
- **Distributed Systems:** "The Bully Algorithm" (Garcia-Molina).
- **Raft:** "In Search of an Understandable Consensus Algorithm".

---

**Day 114 Complete**
