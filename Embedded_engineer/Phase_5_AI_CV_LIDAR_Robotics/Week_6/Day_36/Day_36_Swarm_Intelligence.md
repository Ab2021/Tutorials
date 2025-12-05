# Day 36: Swarm Intelligence & Homogeneous Systems
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 6: Multi-Robot Systems

---

> **📝 Content Creator Instructions:**
> From one robot to many.
> - **Focus:** Boids (Flocking), Stigmergy (Ants), and Homogeneous Swarms.
> - **Code:** Simulating a flock of agents using Reynolds' Rules.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Swarm Intelligence: Complex global behavior emerging from simple local rules.
2.  **Implement** Reynolds' Boids rules: Separation, Alignment, Cohesion.
3.  **Simulate** Stigmergy: Indirect communication via the environment (Virtual Pheromones).
4.  **Analyze** the scalability of homogeneous swarms (O(N) vs O(N^2)).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib scipy
```

### Prior Knowledge
- Vector Math.
- Local vs Global Coordinate Systems.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Emergence

"The whole is greater than the sum of its parts."
*   **Centralized:** A "Brain" commands every robot. (Single point of failure, Computation limit).
*   **Decentralized (Swarm):** Every robot thinks for itself. Behavior emerges.
    *   *Ants:* No General Ant commands the colony. They just follow simple scent trails.
    *   *Robustness:* If 50% of the swarm dies, the mission continues.

### 🔹 Part 2: Reynolds' Boids (1987)

Three rules that create realistic flocking (Birds/Fish):
1.  **Separation:** "Don't crash." Steer away from neighbors if too close.
2.  **Alignment:** "Fly together." Match velocity with neighbors.
3.  **Cohesion:** "Stay close." Steer towards the center of mass of neighbors.

### 🔹 Part 3: Stigmergy

Communication is expensive (Bandwidth). Stigmergy uses the environment.
*   **Pheromones:** Agent A leaves a trail. Agent B detects it later.
*   **Robotics:** Virtual Pheromones (Shared Grid Map). Robot A marks "explored" on a map. Robot B sees it and goes elsewhere.

---

## 💻 Implementation: Boids Simulation

We will simulate 50 boids in 2D.

### 🛠️ Project Structure
```text
day36_swarm/
├── src/
│   ├── boid.py
│   └── flock.py
└── run_sim.py
```

### 👨‍💻 Code Implementation (`src/flock.py`)

```python
import numpy as np
from scipy.spatial import KDTree

class Flock:
    def __init__(self, num_boids=50):
        self.num = num_boids
        self.pos = np.random.rand(num_boids, 2) * 100
        self.vel = (np.random.rand(num_boids, 2) - 0.5) * 2
        self.max_speed = 2.0
        self.perception = 5.0 # View radius
        
    def update(self):
        # Find neighbors efficiently (O(N log N)) using KDTree
        tree = KDTree(self.pos)
        
        forces = np.zeros_like(self.pos)
        
        for i in range(self.num):
            # Query neighbors
            indices = tree.query_ball_point(self.pos[i], self.perception)
            neighbors = [x for x in indices if x != i]
            
            if not neighbors:
                continue
                
            # 1. Separation
            sep = np.zeros(2)
            for n in neighbors:
                diff = self.pos[i] - self.pos[n]
                dist = np.linalg.norm(diff)
                if dist < 2.0: # Too close!
                    sep += diff / (dist + 0.1) # Inverse sq repulsion
                    
            # 2. Alignment
            avg_vel = np.mean(self.vel[neighbors], axis=0)
            align = avg_vel - self.vel[i]
            
            # 3. Cohesion
            center = np.mean(self.pos[neighbors], axis=0)
            coh = center - self.pos[i]
            
            # Weighted Sum
            forces[i] = 1.0 * sep + 0.5 * align + 0.1 * coh
            
        # Integration
        self.vel += forces * 0.1 # dt
        
        # Limit Speed
        speed = np.linalg.norm(self.vel, axis=1)
        mask = speed > self.max_speed
        self.vel[mask] = self.vel[mask] / speed[mask][:,None] * self.max_speed
        
        self.pos += self.vel
        
        # Periodic Boundary (Pacman world)
        self.pos = self.pos % 100
```

### 👨‍💻 Simulation (`run_sim.py`)

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.flock import Flock

flock = Flock(50)

fig, ax = plt.subplots()
scat = ax.scatter(flock.pos[:,0], flock.pos[:,1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

def animate(frame):
    flock.update()
    scat.set_offsets(flock.pos)
    return scat,

ani = FuncAnimation(fig, animate, interval=50)
plt.show()
```

---

## 🔬 Lab Exercise: Area Coverage

### 1. Lab Objectives
- Goal: 10 Robots must cover a $100 \times 100$ area (Lawn mowing).
- **Algorithm:** Lennard-Jones Potential.
    - Robots repel each other up to distance $R$.
    - They fill the space like gas molecules.
- **Metric:** Coverage $\%$ (Voronoi Partition).
- **Observe:** Equilibrium state is a hexagonal lattice packing.

---

## 🚀 Project: "Killer Swarm"

**Scenario:** Drone Light Show.
1.  **Target:** Form a "Star" shape.
2.  **Method:**
    - Each drone has a distinct `target_pos` in the Star shape.
    - Add Boids rules (Separation) to prevent collisions during transition.
    - Result: Smooth, collision-free morphing from random cloud to Star.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Explosion"
*   **Symptom:** Boids fly off screen at infinite speed.
*   **Cause:** Separation force divided by zero (distance=0). Or integration step too large.
*   **Fix:** Cap maximum force. Use `min(dist, epsilon)`.

#### 2. "Clumping"
*   **Symptom:** All boids collapse into a single point.
*   **Cause:** Cohesion too strong. Separation too weak.
*   **Fix:** Tune weights. $W_{sep} > W_{coh}$.

---

## ⚡ Optimization: Spatial Hashing

$O(N^2)$ neighbor search kills swarms > 1000.
**Grid Hash:**
1.  Divide world into grid cells (size = perception radius).
2.  Store boids in hash map: `Map[GridID] -> List[BoidID]`.
3.  Only check neighbors in current and adjacent cells (9 checks).
4.  Complexity becomes $O(N)$.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Homogeneous"?
    *   **A:** All robots are identical (same hardware, same code). Easier to scale.
2.  **Q:** What happens if the leader dies?
    *   **A:** In Boids, there is no leader. Behavior persists. This is "Fault Tolerance".
3.  **Q:** Does Boids guarantee collision avoidance?
    *   **A:** Soft guarantee. If density is too high, forces might be insufficient. For hard guarantee, wrap Boids with ORCA (Day 19).

### Challenge Task
> **Task:** Predator vs Prey.
> 1. Add a "Predator" boid (Red).
> 2. Rule: Predator chases connection. Boids flee Predator (strong repulsion).
> 3. Observe the "Splitting" behaviors of the flock.

---

## 📚 Further Reading
- **Flocks, Herds, and Schools:** Reynolds (SIGGRAPH 1987).
- **Swarm Robotics:** Sahin (2005).

---

**Day 36 Complete**
