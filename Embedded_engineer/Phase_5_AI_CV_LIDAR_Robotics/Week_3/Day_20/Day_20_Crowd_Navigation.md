# Day 20: Crowd Navigation (Social Force)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 3: Advanced Navigation & Planning

---

> **📝 Content Creator Instructions:**
> Navigating among rocks is physics. Navigating among humans is sociology.
> - **Focus:** Social Force Model (SFM), Human-Robot Interaction (HRI), and Handling "The Frozen Robot Problem".
> - **Code:** Implementation of Social Force Model forces in Python.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the Social Force Model: Humans as particles with repulsive semantic fields.
2.  **Differentiate** between standard obstacle avoidance (treating humans as rocks) and social navigation (respecting personal space).
3.  **Implement** a reactive planner that passes on the right and yields to groups.
4.  **Solve** the "Frozen Robot Problem" using Interaction-Aware planning.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Potential Fields.
- Newton's Laws.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Social Force Model (SFM)

Helbing (1995). Humans act as if driven by forces:
$$ F_{total} = F_{goal} + F_{obstacles} + \sum F_{pedestrians} $$

1.  **Goal Force ($F_{goal}$):** "I want to go to target at speed $v_{des}$."
    $$ F_{goal} = m \frac{v_{des} \cdot \vec{e} - v_{curr}}{\tau} $$
2.  **Social Repulsion ($F_{ped}$):** "Get away from me."
    $$ F_{ped} = A \exp(\frac{r_{sum} - d}{B}) \vec{n} $$
    *   $A, B$: Strength and Range of personal space.
    *   $d$: Distance to human.
    *   **Anisotropy:** The force is weaker *behind* the human (we don't have eyes in the back of our heads).

### 🔹 Part 2: The Frozen Robot Problem

In dense crowds, if we treat every human as a hard obstacle with a large inflation radius (Personal Space), the Configuration Space becomes completely blocked.
*   **Result:** The robot stops moving (Freezes) because every path is "Collision".
*   **Solution:** 
    1.  **Cooperative behavior:** Assume humans will make space if the robot slowly pushes forward.
    2.  **Topological Planning:** Identify "Flows" of people and join the flow.

### 🔹 Part 3: Deep Social Navigation

SFM requires tuning $A, B$.
**RL approach:** Train robot in a simulator filled with SFM humans.
*   *Reward:* $+1$ for Goal, $-1$ for Collision, $-0.1$ for entering "Intimate Space" ($<0.5m$).
*   *Learned Behavior:* The robot learns to nudge, weave, and signal intent.

---

## 💻 Implementation: Social Force Simulation

We will simulate a Hallway scenario with 1 Robot and 5 Humans.

### 🛠️ Project Structure
```text
day20_social/
├── src/
│   ├── sfm.py
│   ├── visualizer.py
└── run_crowd.py
```

### 👨‍💻 Code Implementation (`src/sfm.py`)

```python
import numpy as np

class Agent:
    def __init__(self, pos, goal, v_des=1.5, type='human'):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2)
        self.goal = np.array(goal, dtype=float)
        self.v_des = v_des
        self.radius = 0.3
        self.mass = 80 if type == 'human' else 100
        self.type = type

    def update(self, dt, forces):
        acc = forces / self.mass
        self.vel += acc * dt
        
        # Limit speed
        speed = np.linalg.norm(self.vel)
        if speed > 2.0:
            self.vel = self.vel / speed * 2.0
            
        self.pos += self.vel * dt

def compute_sfm(agents):
    forces = {}
    
    for i, a in enumerate(agents):
        f_total = np.zeros(2)
        
        # 1. Goal Force
        diff = a.goal - a.pos
        dist = np.linalg.norm(diff)
        if dist > 0.1:
            dir = diff / dist
            v_wanted = dir * a.v_des
            f_goal = (v_wanted - a.vel) / 0.5 # Tau = 0.5s relaxation
            f_total += f_goal * a.mass
            
        # 2. Social Force (Human-Human / Human-Robot)
        for j, b in enumerate(agents):
            if i == j: continue
            
            diff_p = a.pos - b.pos
            dist_p = np.linalg.norm(diff_p)
            sum_r = a.radius + b.radius
            
            # Simple Repulsion
            # F = A * exp((r - d) / B)
            A = 2000 # Newton
            B = 0.08 # Meter
            
            if dist_p < 5.0: # Optimization radius
                n = diff_p / dist_p
                f_soc = A * np.exp((sum_r - dist_p) / B) * n
                
                # Anisotropy: Reduce force if b is behind a
                # Not implemented for simplicity, but crucial for reality
                
                f_total += f_soc
                
        forces[i] = f_total
        
    return forces
```

### 👨‍💻 Simulation Logic (`run_crowd.py`)

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.sfm import Agent, compute_sfm

# Scenario: Head-on collision
robot = Agent([-5, 0], [5, 0], type='robot')
human = Agent([5, 0.1], [-5, 0.1], type='human') # Slight offset to break symmetry

agents = [robot, human]

def update(frame):
    forces = compute_sfm(agents)
    for i, a in enumerate(agents):
        a.update(0.1, forces[i])
    # ... plotting code ...
```

---

## 🔬 Lab Exercise: The "Excuse Me" Algorithm

### 1. Lab Objectives
- Implement a scenario where the hallway is blocked by two talking humans.
- **Robot strategy:**
    1.  Wait for 3 seconds.
    2.  If blocked, approach slowly to 1.0m.
    3.  If still blocked, approach to 0.5m (invade space) to trigger their repulsion force.
- **Goal:** Observe the humans "move away" due to SFM repulsion from the robot.

### 2. Step-by-Step Guide
1.  Set Human A at $(0, 1)$, Human B at $(0, -1)$. Static ($v_{des}=0$).
2.  Robot goal $(10, 0)$.
3.  Robot Physics: Standard SFM.
4.  Human Physics: If Robot comes close, $F_{soc}$ pushes them aside.

---

## 🚀 Project: "Socially Aware planner (ROS 2)"

**Goal:** Modify `nav2` to respect social norms.
**Idea:** Dynamic Costmap Layer.
1.  **Input:** Detected People (Positions).
2.  **Layer:** Add Gaussian Cost around each person.
    *   **Asymmetric:** Gaussian is longer in *front* of the person (Velocity direction).
    *   Cost = 254 (Lethal) at center.
    *   Cost = 128 (Warning) at 1m front, 0.5m back.
3.  **Planner:** DWB / TEB will naturally plan a path that loops around the "gaze" of the person.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Oscillation" (The Dance)
*   **Symptom:** Robot steps left, Human steps right (same side). Then both switch. Crash.
*   **Cause:** Symmetry.
*   **Fix:** **Right-Hand Rule Bias**. Always prefer passing on the right (modify cost function).

#### 2. "Robot behaves aggressively"
*   **Symptom:** Robot cuts right in front of a walking human.
*   **Cause:** Robot thinks "If I move fast enough, I won't collide." It ignores that this scares the human.
*   **Fix:** Add a cost for **velocity projection intersection**. Do not cross the path of a human within $T=2s$.

---

## ⚡ Optimization: Crowd Simulation

To train RL, we need 1000s of humans.
**PedSim / Mennge:** High-performance crowd simulators compatible with ROS.
*   Can simulate panic, queuing, and grouping behaviors.
*   Use `pedsim_ros` to inject fake people into your lidar topic.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Personal Space" radius for Western culture?
    *   **A:** Intimate (<0.5m), Personal (0.5-1.2m), Social (1.2-3.6m). Robot should stay in Social/Personal, never Intimate.
2.  **Q:** How does SFM handle "Following"?
    *   **A:** By adding an Attractive Force to the leader (or group center).
3.  **Q:** Why is "Freezing" bad?
    *   **A:** It signals submission. In a busy New York street, if you freeze, you will be stuck forever. You must signal intent (move forward) to assert your turn.

### Challenge Task
> **Task:** Implement "Group Detection".
> 1. If two humans define a cluster (distance < 1m) and similar velocity.
> 2. Treat them as a single large obstacle (Polygon).
> 3. Do not try to path plan *between* them (rude!).

---

## 📚 Further Reading
- **Social Force Model:** Helbing and Molnar (Phys. Rev. E 1995).
- **Socially Aware Navigation:** Kretzschmar et al. (ICRA 2016).
- **SEAN:** Social Environment for Autonomous Navigation (Simulator).

---

**Day 20 Complete**
