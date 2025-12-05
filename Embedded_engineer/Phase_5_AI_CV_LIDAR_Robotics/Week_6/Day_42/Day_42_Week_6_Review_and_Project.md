# Day 42: Week 6 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 6: Multi-Robot Systems

---

> **📝 Content Creator Instructions:**
> We have successfully scaled from 1 to N robots.
> - **Goal:** Build a "Amazon Kiva" clone. Fleet of 5 robots moving shelves in a warehouse.
> - **Code:** Integration of Auctions (Task Allocation), CBS (Path Planning), and V2V Comms.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Orchestrate** a fleet of heterogeneous robots to solve a complex logistic problem.
2.  **Mitigate** deadlocks using a hybrid Centralized/Decentralized architecture.
3.  **Simulate** real-world network constraints to test system robustness.
4.  **Visualize** multi-robot traffic flow and efficiency metrics.

---

## 📚 Week 6 Review: The Swarm Stack

| Day | Topic | Key Concept | Use Case |
|-----|-------|-------------|----------|
| **36** | **Swarm** | Local Rules (Boids) | Drones, Search & Rescue |
| **37** | **Coordination** | Consensus / Leader Election | Fault Tolerance |
| **38** | **Map Merging** | Feature Matching / Loop Closure | Cooperative Mapping |
| **39** | **Allocation** | Auctions / CNP | Delivery, Taxi Dispatch |
| **40** | **Formation** | Leader-Follower / Rigidity | Convoy, Large Payload |
| **41** | **Comms** | Mesh / DTN | Caves, Disaster Zones |

### The Architecture Choice
*   **Centralized:** Good for Warehouses (Predictable, High Bandwidth).
*   **Decentralized:** Good for Mars / Caves (Unpredictable, Low Bandwidth).

---

## 🚀 Weekly Capstone: "Warehouse Fleet Manager"

**Scenario:** 5 Robots. $10 \times 10$ Grid. 20 Shelves (Pods) to move to Packing Stations.
**Requirements:**
1.  **Task Allocation:** Simple Auction. Closest robot takes the pod.
2.  **Path Planning:** Conflict-Based Search (CBS) (Day 19) to prevent collisions in narrow aisles.
3.  **Traffic Control:** If CBS fails (timeout), use Traffic Lights at intersections.

### 🛠️ Project Structure
```text
week6_capstone/
├── src/
│   ├── fleet_manager.py
│   ├── robot_agent.py
│   ├── cbs_planner.py
│   └── visualizer.py
└── run_warehouse.py
```

### 👨‍💻 Component 1: Fleet Manager (`src/fleet_manager.py`)

Acts as the "Auctioneer" and "Traffic Controller".

```python
class FleetManager:
    def __init__(self, robots, tasks):
        self.robots = robots
        self.tasks = tasks
        
    def assign_tasks(self):
        # Greedy Assignment (Simplified Auction)
        for task in self.tasks:
            if task.status == 'PENDING':
                best_robot = None
                min_dist = float('inf')
                
                for r in self.robots:
                    if r.status == 'IDLE':
                        d = distance(r.pos, task.pos)
                        if d < min_dist:
                            min_dist = d
                            best_robot = r
                            
                if best_robot:
                    best_robot.assign(task)
                    task.status = 'ASSIGNED'
                    
    def plan_paths(self):
        # Run CBS for all active robots
        active_robots = [r for r in self.robots if r.status == 'BUSY']
        starts = [r.pos for r in active_robots]
        goals = [r.task.destination for r in active_robots]
        
        # Solving MAPF
        paths = cbs_solve(starts, goals) 
        
        # Dispatch paths to robots via "Network"
        for i, r in enumerate(active_robots):
            r.set_path(paths[i])
```

### 👨‍💻 Component 2: Robot Agent (`src/robot_agent.py`)

Simulates the robot's local execution and communication dropouts.

```python
class RobotAgent:
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.path = []
        self.status = 'IDLE'
        self.battery = 100.0
        
    def step(self, network_status):
        if self.path:
            # Check "Heartbeat" logic
            if network_status == 'DISCONNECTED':
                # Safe stop? Or Continue blindly?
                # Warehouse rule: If comms lost, STOP immediately.
                return
            
            next_pos = self.path.pop(0)
            self.move_to(next_pos)
            
            if not self.path:
                self.status = 'IDLE' # Task Done
                
    def move_to(self, pos):
        # Simulate kinematics
        self.pos = pos
        self.battery -= 0.1
```

### 👨‍💻 Component 3: The Chaos Simulation (`run_warehouse.py`)

We inject failures to test robustness.

```python
import random

def simulate():
    manager = FleetManager(...)
    
    for t in range(1000):
        # 1. Random Wifi Dropout
        net_status = 'CONNECTED'
        if random.random() < 0.05: # 5% chance
            net_status = 'DISCONNECTED'
            print("⚠️ NETWORK FAILURE")
            
        # 2. Manager Logic
        if net_status == 'CONNECTED':
            manager.assign_tasks()
            manager.plan_paths()
            
        # 3. Robot Logic
        for r in manager.robots:
            r.step(net_status)
            
        # 4. Visualization
        visualizer.draw(manager.robots)
```

---

## 📝 Self-Assessment Quiz

1.  **Architecture:**
    *   Why did we use Centralized logic (Manager) for the Warehouse?
        *   **A:** Because efficiency (Throughput) is the #1 metric in warehouses. Centralized Planning (CBS) is optimal. Decentralized (ORCA) is suboptimal and can deadlock in tight corridors.

2.  **Scalability:**
    *   What happens if we have 1000 robots?
        *   **A:** CBS (NP-Hard) will timeout. Start partitioning the warehouse into "Zones", or switch to a Highway System (Directed Graphs) where robots only flow one way.

3.  **Heterogeneity:**
    *   Could we add a Forklift robot?
        *   **A:** Yes, the Auction system handles it easily. Forklift bids $\infty$ for "Small Box" tasks, and bids Low for "Pallet" tasks.

---

## ⏭️ Look Ahead: Week 7
Robots are working together. Now they must work with **Humans**.
**Week 7: Human-Robot Interaction (HRI).**
*   Voice commands (NLP).
*   Gesture recognition.
*   Safety standards (ISO 13482).
*   Ethics.

---

**Week 6 Complete**
