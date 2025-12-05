# Day 119: Week 17 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 17: Swarm Robotics

---

> **📝 Content Creator Instructions:**
> Order from Chaos.
> - **Goal:** Integrate Discovery, Consensus, and Formation Control into a unified Swarm Stack.
> - **Code:** A "Warehouse Sim" where robots autonomously organize to transport simulated payloads, handling collisions and network drops.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Decompose** a swarm mission into Global Tasks (Dispatch) and Local Tasks (Avoidance).
2.  **Deploy** a heterogeneous swarm (one Leader, many Workers).
3.  **Recover** from partial system failure (Leader death).
4.  **Visualize** state of 20 robots simultaneously.

---

## 📚 Week 17 Review: The Power of Many

| Day | Topic | Key Lesson | Tool |
|-----|-------|------------|------|
| **113** | **Architecture** | Scalability via Namespaces & Discovery | `ros2 launch` |
| **114** | **Consensus** | Agreement in distributed systems | `Bully Algorithm` |
| **115** | **Formation** | Organic movement via Vectors | `Boids` |
| **116** | **Map Merging** | fusing distributed perceptions | `OccupancyGrid` |
| **117** | **Traffic** | Resource negotiation | `TrafficManager` |
| **118** | **Simulation** | ECS & Performance tuning | `Ignition Gazebo` |

### The "Hive Mind" Diagram
```mermaid
graph TD
    subgraph Robot 1
    Sensors1 --> Perception1
    Perception1 --> LocalMap1
    LocalMap1 --> MapMerge
    LocalMap1 --> Avoidance1
    
    Consensus1[Consensus Node] -->|Leader?| Strategy1
    Strategy1 -->|High Level| Dispatch1
    Dispatch1 -->|Waypoints| Navigation1
    Avoidance1 -->|Repulsion| Navigation1
    end
    
    subgraph Robot 2
    Sensors2 --> Perception2
    Perception2 --> LocalMap2
    LocalMap2 --> MapMerge
    
    Consensus2 -->|Leader?| Strategy2
    end
    
    MapMerge -->|Global Map| Strategy2
    Robot 1 -.->|Heartbeat| Robot 2
    Robot 2 -.->|Heartbeat| Robot 1
```

---

## 🚀 Weekly Capstone: "The Warehouse Floor"

**Scenario:** 10 Robots. 10 Packages at Zone A. Move to Zone B.
**Constraints:**
1.  **Corridor:** Only 2 robots wide.
2.  **Traffic:** Must use `TrafficManager` for the corridor.
3.  **Leadership:** If the Manager node dies, a Robot must spawn a local Traffic Manager and broadcast "I am the Captain".

### 🛠️ Project Structure
```text
week17_capstone/
├── src/
│   ├── fleet_commander.py
│   ├── worker_drone.py
│   └── dynamic_traffic.py
├── launch/
│   └── warehouse_sim.launch.py
└── config/
    └── behavior_tree.xml
```

### 👨‍💻 Logic Flow

1.  **Startup:**
    *   `warehouse_sim.launch.py` spawns 10 bots.
    *   Robots perform **Discovery** (Day 113).
    *   Robots perform **Election** (Day 114). Robot_9 wins.
2.  **Mission:**
    *   Robot_9 (Leader) reads `orders.json` (10 packages).
    *   Robot_9 assigns Package 1 to Robot_0 via `/swarm/task`.
3.  **Execution:**
    *   Robot_0 navigates to Pickup.
    *   Robot_0 navigates to Corridor.
    *   Robot_0 requests **Lock** (Day 117).
    *   Leader Grants Lock.
    *   Robot_0 crosses.
4.  **Failure Event:**
    *   Simulation kills Robot_9.
    *   Swarm detects timeout.
    *   New Election. Robot_8 wins.
    *   Robot_8 checks "Completed Tasks" (replicated state) and re-issues pending tasks.

### 👨‍💻 Behavior Tree (XML)

Using `BehaviorTree.CPP` or `py_trees` is better than if/else for this.

```xml
<root main_tree_to_execute = "MainTree">
    <BehaviorTree ID="MainTree">
        <Sequence name="Mission">
            <Fallback name="Connect">
                 <Condition ID="IsConnected"/>
                 <Action ID="FindNeighbors"/>
            </Fallback>
            <Fallback name="Role">
                 <Condition ID="IsLeader"/>
                 <Action ID="FollowOrders"/>
            </Fallback>
            <Sequence name="Lead">
                 <Action ID="AssignTasks"/>
                 <Action ID="ManageTraffic"/>
            </Sequence>
        </Sequence>
    </BehaviorTree>
</root>
```

---

## 📝 Self-Assessment Quiz

1.  **Scalability:**
    *   Why does "All-to-All" communication fail at $N=100$?
    *   **A:** Network congestion scales with $N^2$. Packet loss rises, latency spikes, heartbeat timeouts trigger false elections.
2.  **Simulation:**
    *   What is "Real Time Factor"?
    *   **A:** RTF = Sim Time / Wall Time. If RTF = 0.5, 1 second of simulation takes 2 seconds of real time.
3.  **Map Merging:**
    *   Do we merge maps continuously?
    *   **A:** Ideally no. Merge once to build the global map, then share small updates (dynamic obstacles).

---

## ⏭️ Look Ahead: Week 18
From Land to **Air and Sea**.
**Week 18: Aerial & Underwater Robotics.**
*   The physics change (6-DOF, Gravity, Buoyancy).
*   Flight Controllers (PX4).
*   Visual Inertial Odometry (VIO) becomes critical (No wheel encoders!).
*   Sonar for underwater sensing.

---

**Week 17 Complete**
