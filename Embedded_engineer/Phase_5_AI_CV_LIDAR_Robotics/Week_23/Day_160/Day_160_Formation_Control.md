# Day 160: Decentralized Formation Control
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 23: V2X & Swarm Intelligence

---

> **📝 Content Creator Instructions:**
> Fly like a flock.
> - **Focus:** Graph Theory (Laplacian Matrix), Consensus Protocols ($\dot{x}_i = \sum a_{ij}(x_j - x_i - d_{ji})$), Virtual Structure vs Leader-Follower, and Handling topology changes.
> - **Code:** A Python script `formation_fly.py` simulating 5 drones. They form a Pentagon. When the user drags the "Virtual Leader", the formation deforms and restores itself using neighbor-only comms.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Construct** the Laplacian Matrix $L$ for a communication graph.
2.  **Implement** the Consensus Protocol for Agreement ($x_i \to x_{avg}$).
3.  **Shape** the formation using Relative Offsets ($Bias_{ij}$).
4.  **Simulate** formation flight where agents only talk to neighbors.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib networkx
```

### Prior Knowledge
- Linear Algebra (Eigenvalues).
- Differential Equations.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Consensus Problem

How do 100 drones agree on a height $H$ without a central commander?
Each drone looks at its neighbors $N_i$.
$$ \dot{x}_i = -k \sum_{j \in N_i} (x_i - x_j) $$
*   They move towards the average of their neighbors.
*   Eventually, $x_1 = x_2 = \dots = x_N$. Convergence guaranteed if Graph is Connected.

### 🔹 Part 2: Formation Geometry

We don't want them to collide at the same point. We want a shape.
$$ \dot{x}_i = -k \sum_{j \in N_i} ((x_i - h_i) - (x_j - h_j)) $$
*   $h_i$: Desired offset of Agent $i$ from the Virtual Center.
*   Essentially, they run consensus on the "Virtual Center" estimate.

### 🔹 Part 3: Graph Laplacian

The dynamics of the whole system:
$$ \dot{X} = -L X $$
*   $L = D - A$ (Degree Matrix - Adjacency Matrix).
*   Eigenvalues of $L$ determine convergence speed ($Re(\lambda_2)$ is algebraic connectivity).

---

## 💻 Implementation: The Flying Pentagon

We simulate 5 agents in 2D. They communicate in a ring topology (0-1-2-3-4-0).

### 🛠️ Project Structure
```text
day160_formation/
├── src/
│   ├── formation_fly.py
└── output/
    ├── formation_plot.png
```

### 👨‍💻 Formation Sim (`src/formation_fly.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Agent:
    def __init__(self, id, pos, desired_offset):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.desired_offset = np.array(desired_offset, dtype=float)
        self.vel = np.zeros(2)
        self.neighbors = [] # List of references to other agents

    def compute_control(self):
        # Consensus Protocol with Bias
        # u_i = sum( (pos_j - offset_j) - (pos_i - offset_i) )
        
        force = np.zeros(2)
        k_p = 2.0
        
        # Virtual Center Estimation error
        my_center_est = self.pos - self.desired_offset
        
        for neighbor in self.neighbors:
            neighbor_center_est = neighbor.pos - neighbor.desired_offset
            
            # Pull towards neighbor's estimate of center
            force += (neighbor_center_est - my_center_est)
            
        self.vel = k_p * force
        return self.vel

    def update(self, dt):
        self.pos += self.vel * dt

def main():
    num_agents = 5
    radius = 10.0
    
    # Define Target Shape (Pentagon)
    offsets = []
    for i in range(num_agents):
        angle = 2 * np.pi * i / num_agents
        offsets.append([radius * np.cos(angle), radius * np.sin(angle)])
        
    # Initialize Agents at random positions
    agents = []
    for i in range(num_agents):
        start_pos = np.random.uniform(-30, 30, 2)
        agents.append(Agent(i, start_pos, offsets[i]))
        
    # Define Topology (Ring)
    # 0<->1, 1<->2, ... 4<->0
    for i in range(num_agents):
        next_idx = (i + 1) % num_agents
        prev_idx = (i - 1 + num_agents) % num_agents
        
        # We need actual references after list is full
        pass

    # Connect Neighbors
    for i in range(num_agents):
        next_idx = (i + 1) % num_agents
        prev_idx = (i - 1 + num_agents) % num_agents
        agents[i].neighbors.append(agents[next_idx])
        agents[i].neighbors.append(agents[prev_idx])

    # Simulation
    history_x = {i: [] for i in range(num_agents)}
    history_y = {i: [] for i in range(num_agents)}
    
    # Introduce a "Leader" drift?
    # Let's say Agent 0 sees a Mouse (Target) at (50, 50) and adds a term to its control
    # Or just let them converge to average of start positions (Consensus)
    
    print("Simulating Formation Convergence...")
    
    for t in range(200): # 200 steps
        dt = 0.05
        
        # Calculate all controls first (Synchronous)
        controls = [a.compute_control() for a in agents]
        
        # Apply Lead term to Agent 0 (Drag the formation)
        if t > 50:
             # Agent 0 wants to go to (40, 40) relative to center?
             # No, simply add velocity vector
             controls[0] += np.array([1.0, 0.5]) * 5.0 # External input
        
        # Apply
        for i, a in enumerate(agents):
            a.vel = controls[i]
            a.update(dt)
            
            history_x[i].append(a.pos[0])
            history_y[i].append(a.pos[1])
            
    # Visualization
    plt.figure(figsize=(10, 10))
    for i in range(num_agents):
        plt.plot(history_x[i], history_y[i], label=f'Agent {i}')
        # Plot final pos
        plt.plot(history_x[i][-1], history_y[i][-1], 'o', markersize=10)
        
    # Draw connections at end
    final_x = [history_x[i][-1] for i in range(num_agents)]
    final_y = [history_y[i][-1] for i in range(num_agents)]
    # Connect ring
    final_x.append(final_x[0])
    final_y.append(final_y[0])
    plt.plot(final_x, final_y, 'k--', alpha=0.5, label='Links')
    
    plt.title("Decentralized Formation Control (Ring Topology)")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.savefig("output/formation_plot.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Broken Link"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Formation converges to a Pentagon and moves North-East (dragged by Agent 0).
- **Modify:** Cut the link between Agent 2 and 3. `agents[2].neighbors.remove(...)`.
- **Result:** Line Topology vs Ring. Convergence is slower ($\lambda_2$ decreases).
- **Fail:** Cut Agent 0 from everyone.
- **Result:** Agent 0 flies away. Rest stay behind. Formation breaks.

---

## 🚀 Project: "Reconfiguration"

**Goal:** Shape Shifting.
1.  **State 1:** Pentagon (Radius 10).
2.  **Trigger:** Pass through narrow door.
3.  **State 2:** Change `desired_offset` to form a Line (Column).
4.  **Dynamics:** Agents smoothly transition to new slots without collision (if potential fields added).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Oscillation"
*   **Cause:** $k_p$ too high or Damping $k_d$ missing.
*   **Fix:** Add damping term: $-k_d (v_i - v_j)$ relative velocity damping.

#### 2. "Rigid Body Rotation"
*   **Cause:** The Consensus law defined above controls position relative to center, but doesn't constrain Rotation of the whole group.
*   **Result:** The Pentagon might spin uniformly.
*   **Fix:** Share Heading info $\theta_i$ or use absolute coordinates (Compass).

---

## ⚡ Optimization: Event-Triggered Control

Communication consumes battery.
*   **Method:** Only broadcast $Position$ if it has changed by $> \delta$ since last broadcast.
*   **Result:** Comms reduced by 90% when flying steady, high comms during aggressive turns.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Algebraic Connectivity"?
    *   **A:** The second smallest eigenvalue of the Laplacian ($\lambda_2$). Higher = Faster convergence / Stronger network.
2.  **Q:** Leader-Follower vs Virtual Structure?
    *   **A:** L-F: Error propagates (Tail wags). VS: Everyone tracks a virtual point, error is distributed evenly.
3.  **Q:** How to avoid collision within formation?
    *   **A:** Artificial Potentials (Repulsion 1/r) added to the consensus control law.

### Challenge Task
> **Task:** 3D Drone Show.
> 1. Extend to $z$ axis.
> 2. Create the "Intel/Olympics" drone show logic.
> 3. Use pre-calculated trajectories (Centralized) vs Consensus (Decentralized)?
> 4. (Answer: Shows are Centralized for safety/precision. Swarms are Decentralized for robustness).

---

## 📚 Further Reading
- **Mesbahi & Egerstedt:** "Graph Theoretic Methods in Multiagent Networks".
- **Reynolds:** "Boids" (Flocking behavior).

---

**Day 160 Complete**
