# Day 39: Task Allocation (Auction Algorithms)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 6: Multi-Robot Systems

---

> **📝 Content Creator Instructions:**
> We have 10 robots and 100 packages. Who picks up what?
> - **Focus:** Market-Based Allocation, Contract Net Protocol (CNP), and Handling Failures.
> - **Code:** Simulating a bidding system for a delivery fleet.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Formulate** the Multi-Robot Task Allocation (MRTA) problem as an optimization problem.
2.  **Implement** a Sealed-Bid Auction system.
3.  **Explain** the Contract Net Protocol (CNP) workflow: Announce -> Bid -> Award.
4.  **Handle** "Stranded Assets" (Winner dies before completing task) using Re-Auctioning.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
# Standard Python
```

### Prior Knowledge
- Optimization (Minimizing Cost).
- Finite State Machines.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: MRTA Taxonomy

Gerkey & Mataric Taxonomy (2004):
1.  **ST-SR (Single-Task, Single-Robot):** One robot does one job. Simple.
2.  **ST-MR (Single-Task, Multi-Robot):** Box is too heavy for one robot. Need 2 to lift. (Coalition).
3.  **MT-SR (Multi-Task, Single-Robot):** Robot must visit points A, B, C. (TSP - Traveling Salesman).
4.  **MT-MR (Multi-Task, Multi-Robot):** The Holy Grail. (mTSP).

### 🔹 Part 2: Market-Based Approaches

Robots act as selfish economic agents.
*   **Cost Function:** Battery Power + Distance + Workload.
*   **Revenue:** Task Reward.
*   **Profit:** Revenue - Cost.
*   **Goal:** Maximize own profit $\implies$ Maximizes fleet efficiency (Invisible Hand).

### 🔹 Part 3: Contract Net Protocol (CNP)

Standard FIPA protocol:
1.  **Manager** broadcasts: "Task $T$ available at $(x,y)$."
2.  **Contractors (Robots)** compute estimate $C_i$.
3.  **Contractors** send Bid: "I can do it for $10."
4.  **Manager** selects lowest bid ($min(C_i)$). Sends "AWARD".
5.  **Winner** sends "ACCEPT" or "REJECT" (if status changed).

---

## 💻 Implementation: Delivery Bider

Scenario: 3 Robots, 5 Packages appearing randomly.

### 🛠️ Project Structure
```text
day39_auction/
├── src/
│   ├── auctioneer.py
│   ├── bidder.py
│   └── task.py
└── run_market.py
```

### 👨‍💻 Code Implementation (`src/bidder.py`)

```python
import numpy as np

class RobotBidder:
    def __init__(self, id, start_pos, speed=1.0):
        self.id = id
        self.pos = np.array(start_pos)
        self.speed = speed
        self.tasks = [] # Queue of assigned tasks
        
    def calculate_bid(self, task_pos):
        # Cost = Travel Time + Wait Time (existing queue)
        
        # 1. Travel from current pos to last queued task
        if self.tasks:
            last_pos = self.tasks[-1]['pos']
            current_queue_time = self.calculate_queue_time()
        else:
            last_pos = self.pos
            current_queue_time = 0
            
        # 2. Travel from last pos to new task
        dist = np.linalg.norm(np.array(task_pos) - np.array(last_pos))
        travel_time = dist / self.speed
        
        # Total cost (Price)
        bid_price = current_queue_time + travel_time
        return bid_price

    def calculate_queue_time(self):
        # Sum of travel times in queue
        t = 0
        curr = self.pos
        for task in self.tasks:
            dist = np.linalg.norm(np.array(task['pos']) - np.array(curr))
            t += dist / self.speed
            curr = task['pos']
        return t
        
    def award_task(self, task):
        self.tasks.append(task)
        print(f"Robot {self.id} won task {task['id']}!")
```

### 👨‍💻 Simulation (`run_market.py`)

```python
from src.bidder import RobotBidder

# Fleet
r1 = RobotBidder(1, [0, 0], speed=1.0)
r2 = RobotBidder(2, [10, 10], speed=2.0) # Faster but further away
fleet = [r1, r2]

# Tasks appearing
tasks = [
    {'id': 'A', 'pos': [2, 2]},
    {'id': 'B', 'pos': [9, 9]},
    {'id': 'C', 'pos': [0, 1]}
]

print("--- Start Auction ---")

for task in tasks:
    print(f"\nAuctioning Task {task['id']} at {task['pos']}")
    best_bid = float('inf')
    winner = None
    
    for robot in fleet:
        bid = robot.calculate_bid(task['pos'])
        print(f"  Robot {robot.id} bids: {bid:.2f}")
        
        if bid < best_bid:
            best_bid = bid
            winner = robot
            
    if winner:
        winner.award_task(task)

print("\n--- Final Allocation ---")
print(f"Robot 1 Queue: {[t['id'] for t in r1.tasks]}")
print(f"Robot 2 Queue: {[t['id'] for t in r2.tasks]}")
```

### 3. Expected Output
*   Task A (2,2): Robot 1 is closer (Bid ~2.8), Robot 2 (Bid ~6). Robot 1 wins.
*   Task B (9,9): Robot 1 is busy (Cost adds to A). Robot 2 is closer. Robot 2 wins.
*   Task C (0,1): Robot 1 is very close to A. Even with queue, it might win.

---

## 🔬 Lab Exercise: The Greedy Fail

### 1. Lab Objectives
- **Scenario:** Points on a line: `R1 -- A -- B -- R2`.
- **Greedy Auction:** R1 takes A. R2 takes B.
- **Problem:** If R1 is huge and slow, and R2 is tiny and fast, maybe R2 should have taken both.
- **Observation:** Sequential Single-Item Auctions (SSI) are suboptimal compared to Combinatorial Auctions (bundles).
- **Fix:** Allow robots to bid on *bundles* of tasks {A, B}. (Computationally expensive).

---

## 🚀 Project: "Search and Rescue"

**Scenario:** Fire in a building.
1.  **Central Controller:** Identifies "Zones" that need checking.
2.  **Robots:** Heterogeneous.
    *   UAVs (Fast, Low Battery).
    *   UGVs (Slow, High Battery, Can carry water).
3.  **Cost Function:**
    *   UAV Bid: `Time` (Low) but `Capability` (Cannot extinguish fire).
    *   UGV Bid: `Time` (High) but `Capability` (Can extinguish).
4.  **Result:** UAVs win "Scouting" tasks. UGVs win "Extinguish" tasks.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "The Idle Robot"
*   **Symptom:** R1 wins everything because it's slightly faster. R2 sits idle. Global efficiency suffers because R1 has a 10-hour queue.
*   **Fix:** **Load Balancing**. Add a penalty to the cost function for `QueueLength`, effectively dynamically raising the price as the robot gets busier.

#### 2. "Task Starvation"
*   **Symptom:** A hard task (far away) never gets picked because bids are too high.
*   **Fix:** **Reserve Price** or Forced Assignment if no one bids below threshold.

---

## ⚡ Optimization: Distributed Constraint Optimization (DCOP)

Auctions are greedy heuristics. DCOP is formal optimization.
*   Algorithms like **Max-Sum** allow agents to negotiate and swap tasks to reach global optimality.
*   Allows "I'll trade you Task A for Task B if you give me $5."

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Vickrey Auction"?
    *   **A:** Second-price sealed-bid auction. Winner pays the price of the *second* highest bid. Encourages truthful bidding (bidding your true value).
2.  **Q:** How to handle robot death?
    *   **A:** Heartbeats. If winner stops pinging, the Task Manager puts the task back into the Auction Pool (Re-auction).
3.  **Q:** Can we auction computation?
    *   **A:** Yes! Cloud Robotics. "Who has free GPU cycles to run my heavy V-SLAM?"

### Challenge Task
> **Task:** Dynamic Re-Allocation.
> 1. Robot 1 wins Task A.
> 2. While moving, Robot 1 gets a flat tire (Speed drops).
> 3. Robot 1 must *sell* Task A back to the market logic.

---

## 📚 Further Reading
- **Contract Net Protocol:** Smith (1980).
- **MRTA Taxonomy:** Gerkey & Mataric (IJRR 2004).

---

**Day 39 Complete**
