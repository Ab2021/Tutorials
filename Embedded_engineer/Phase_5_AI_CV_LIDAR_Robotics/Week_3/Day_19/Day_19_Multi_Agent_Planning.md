# Day 19: Multi-Agent Path Finding (MAPF)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 3: Advanced Navigation & Planning

---

> **📝 Content Creator Instructions:**
> One robot is easy. Ten robots is chaos.
> - **Focus:** Conflict-Based Search (CBS) for optimality, and ORCA (Velocity Obstacles) for decentralized avoidance.
> - **Code:** Implementation of CBS for grid-based multi-robot planning.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Centralized (CBS, Prioritized Planning) and Decentralized (ORCA, D-RRT) approaches.
2.  **Formulate** MAPF as finding a joint path $\Pi = (\pi_1, ..., \pi_N)$ that minimizes flowtime summation.
3.  **Implement** Conflict-Based Search (CBS) to solve multi-agent grid problems optimally.
4.  **Simulate** swarm behavior using RVO2 (Reciprocal Velocity Obstacles).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install networkx numpy matplotlib
# Optional: RVO2 Python library
```

### Prior Knowledge
- A* Search (Space-Time A*).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Curse of Dimensionality

If one robot has state space $S$, $N$ robots have state space $S^N$.
*   A* on $10 \times 10$ grid for 1 robot: $100$ states.
*   A* on $10 \times 10$ grid for 10 robots: $100^{10}$ states.
*   **Result:** Joint A* is intractable. We need decoupled approaches.

### 🔹 Part 2: Conflict-Based Search (CBS)

A two-level algorithm:
1.  **High Level:** Search the "Constraint Tree". Nodes are constraints (e.g., "Agent 1 cannot be at $(x,y)$ at time $t$").
2.  **Low Level:** Plan path for *single* agent consistent with constraints (using A*).

**Algorithm:**
1.  Plan individual optimal paths ignoring others.
2.  Check for **Conflict** (Robot A and B share same cell at same time).
3.  If Conflict $(A, B, v, t)$ found:
    *   Split into two child nodes.
    *   Child 1: Add Constraint: Agent A cannot use $v$ at $t$. Re-plan A.
    *   Child 2: Add Constraint: Agent B cannot use $v$ at $t$. Re-plan B.
4.  Repeat until no conflicts.

### 🔹 Part 3: Decentralized Avoidance (ORCA)

**Velocity Obstacle (VO):**
The set of relative velocities that will result in a collision at some future time $\tau$.
**ORCA (Optimal Reciprocal Collision Avoidance):**
*   Assumption: The other robot will *also* try to avoid me.
*   Action: I take half the responsibility. I shift my velocity by $u/2$, and assume they shift by $-u/2$.
*   *Result:* Smooth, oscillation-free avoidance for thousands of agents (used in video games/crowd sims).

---

## 💻 Implementation: CBS from Scratch

We will solve a classic Grid MAPF problem.

### 🛠️ Project Structure
```text
day19_mapf/
├── src/
│   ├── cbs.py
│   ├── a_star.py
│   └── visualizer.py
└── run_demo.py
```

### 👨‍💻 Code Implementation (`src/cbs.py`)

```python
import heapq
from src.a_star import a_star

class Constraints:
    def __init__(self):
        self.constraints = [] # List of {'agent': id, 'loc': [x,y], 'timestep': t}

    def add(self, constraint):
        self.constraints.append(constraint)

    def is_constrained(self, agent_id, loc, timestep):
        for c in self.constraints:
            if c['agent'] == agent_id and c['loc'] == loc and c['timestep'] == timestep:
                return True
        return False

class CBSNode:
    def __init__(self):
        self.constraints = Constraints()
        self.paths = {} # agent_id -> path
        self.cost = 0

    def get_first_conflict(self, agents):
        # Check all pairs
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                path_i = self.paths[i]
                path_j = self.paths[j]
                
                # Check overlap at same time
                min_len = min(len(path_i), len(path_j))
                for t in range(min_len):
                    if path_i[t] == path_j[t]:
                        return {'a1': i, 'a2': j, 'loc': path_i[t], 'timestep': t}
        return None

class CBS:
    def __init__(self, grid, agents):
        self.grid = grid
        self.agents = agents # List of {'start':, 'goal':}

    def solve(self):
        root = CBSNode()
        # Initial Plan
        for i, agent in enumerate(self.agents):
            path = a_star(self.grid, agent['start'], agent['goal'], Constraints(), i)
            if path is None: return None
            root.paths[i] = path
            root.cost += len(path)

        open_list = []
        heapq.heappush(open_list, (root.cost, id(root), root))

        while open_list:
            _, _, curr_node = heapq.heappop(open_list)
            
            conflict = curr_node.get_first_conflict(self.agents)
            if not conflict:
                return curr_node.paths # Solution!

            # Branching
            # Child 1: Constrain Agent 1
            child1 = self.create_child(curr_node, conflict['a1'], conflict, conflict['a1'])
            if child1: heapq.heappush(open_list, (child1.cost, id(child1), child1))
            
            # Child 2: Constrain Agent 2
            child2 = self.create_child(curr_node, conflict['a2'], conflict, conflict['a2'])
            if child2: heapq.heappush(open_list, (child2.cost, id(child2), child2))
            
        return None
```

---

## 🔬 Lab Exercise: Warehouse Logistics

### 1. Lab Objectives
- Simulate an "Amazon Kiva" warehouse scenario.
- 5 Robots, 5 Pods, narrow corridors.
- **Task:** Robots must swap positions (Corridor Logic).
- **Observation:** CBS figures out that one robot must wait in a nook while the other passes.

### 2. Step-by-Step Guide
1.  Define Grid $(10 \times 10)$.
2.  Agents: A start $(0,0) \to (0,9)$. B start $(0,9) \to (0,0)$.
3.  Run CBS.
4.  Visualize: Agent A moves to $(1,5)$ (sidetrack), waits for B to pass, then resumes.

---

## 🚀 Project: "Drone Swarm Formation"

**Goal:** 10 Drones fly from random start to a "V" formation.
**Method:** decentralized ORCA (`rvo2` library).
1.  Assign target position in "V" shape to each drone.
2.  Update Velocities using RVO.
3.  Visualize in 3D (Matplotlib / Gazebo).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Livin' on the Edge" (Vertex vs Edge Conflict)
*   **Problem:** Robots swap positions: $A: (0,0) \to (0,1)$, $B: (0,1) \to (0,0)$.
*   **Result:** They crash in the middle (Edge Conflict).
*   **Fix:** CBS must check for **Edge Constraints** (Traversing edge $u \to v$ at $t$) in addition to Vertex Constraints.

#### 2. "Timeout"
*   **Problem:** CBS is Exponential in worst case.
*   **Fix:** Use **ECBS** (Suboptimal CBS). Use a focal search with weight $w$. Allows paths that are $\le w \times Optimal$ but solves much faster.

---

## ⚡ Optimization: Geometric Containers

For massive swarms ($>100$), don't plan individual paths.
**Swarm Control:** Treat swarm as a fluid or shape.
*   Control the boundary of the shape.
*   Robots fill the shape uniformly using local repulsion forces.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just Prioritized Planning (Plan A, then Plan B avoiding A)?
    *   **A:** It is incomplete. A might block B's goal forever. CBS re-plans A if necessary.
2.  **Q:** What is the "Reciprocal" in ORCA?
    *   **A:** It means Robot A assumes Robot B will do half the work. Without this, A does all the work, or both do double work (oscillation).

### Challenge Task
> **Task:** Implement Traffic Rules.
> 1. Use Decoupled Planning (Path A, Path B).
> 2. Robots must yield to the one on the Right.
> 3. Does this resolve deadlocks? (Sometimes, but circular deadlocks still exist).

---

## 📚 Further Reading
- **CBS:** Sharon et al. (AAAI 2012).
- **ORCA:** Van den Berg et al. (ISRR 2011).
- **Amazon Robotics:** "Multi-Agent Path Finding for Kiva Systems".

---

**Day 19 Complete**
