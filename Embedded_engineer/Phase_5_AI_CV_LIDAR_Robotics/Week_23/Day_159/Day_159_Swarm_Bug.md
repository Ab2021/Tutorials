# Day 159: Swarm Gradient Bug Algorithm
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 23: V2X & Swarm Intelligence

---

> **📝 Content Creator Instructions:**
> Ants don't use GPS. They follow the scent.
> - **Focus:** Swarm Source Seeking (Gradient Descent), Handling Map Minima, The Bug Algorithm ($Bug 0, Bug 1, Bug 2$) for obstacle circumnavigation.
> - **Code:** A Python script `swarm_bug.py` simulating 20 robots. They seek a "Light Source" (Target). When they hit a wall, they execute a Bug Algorithm (Wall Follow) until the gradient improves.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** Gradient Descent navigation ($v \propto \nabla Signal$).
2.  **Code** a Finite State Machine for Bug behavior: `SEEK` $\to$ `HIT` $\to$ `FOLLOW` $\to$ `LEAVE`.
3.  **Simulate** a Swarm of agents operating independently without a central map.
4.  **Compare** Bug 1 (Full Circumnavigation) vs Bug 2 (M-Line Departure).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Gradient Vectors.
- State Machines.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Source Seeking

Goal: Find the source of a signal $J(x,y)$ (e.g., Radiation, Light, Chemical).
*   **Strategy:** Move uphill.
    $$ \vec{v} = k \cdot \nabla J(x,y) $$
*   **Problem:** Walls. Local Maxima.

### 🔹 Part 2: The Bug Algorthms

How to bypass an obstacle without a map?
*   **Bug 0:** Follow wall until head points to target. (Unsafe, loops forever in C-shapes).
*   **Bug 1:** Follow wall for full revolution. Record point closest to target. Go there. Leave wall. (Safe, but slow).
*   **Bug 2:** Define Line $L$ from Start to Target (M-Line).
    1.  Move towards Target.
    2.  Hit Wall. Follow Wall.
    3.  If you cross M-Line AND Distance < Distance_at_Hit, Leave Wall.

---

## 💻 Implementation: The Light Seekers

We simulate 20 robots in a messy room.

### 🛠️ Project Structure
```text
day159_swarm/
├── src/
│   ├── swarm_bug.py
└── output/
    ├── swarm_viz.png
```

### 👨‍💻 Swarm Bug Simulator (`src/swarm_bug.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.target = np.array([80.0, 80.0])
        self.obstacles = [
            {'x': 50, 'y': 50, 'r': 15}, # Center Obstacle
            {'x': 20, 'y': 60, 'r': 10},
            {'x': 70, 'y': 30, 'r': 10}
        ]
        
    def get_signal(self, pos):
        # Inverse Square Law
        dist = np.linalg.norm(pos - self.target)
        return 1000.0 / (dist + 1.0)
        
    def check_collision(self, pos):
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - np.array([obs['x'], obs['y']]))
            if dist < obs['r']:
                return True, np.array([obs['x'], obs['y']])
        return False, None

class Robot:
    def __init__(self, id, start_pos, target_pos, env):
        self.id = id
        self.pos = np.array(start_pos, dtype=float)
        self.target = np.array(target_pos, dtype=float)
        self.env = env
        
        self.state = "SEEK" # SEEK, WALL_FOLLOW
        self.hit_point = None
        self.dist_at_hit = 0.0
        
        # M-Line (Start to Target)
        # Vector form: P = Start + t * (Target - Start)
        # We check distance to this line segment for Bug 2, or just the line logic
        # Bug 2 simplified check: Are we ON the line connecting Start-Target?
        self.start_pos_fixed = np.array(start_pos, dtype=float)
        
    def get_gradient_dir(self):
        # Normalize vector to target
        diff = self.target - self.pos
        norm = np.linalg.norm(diff)
        if norm < 0.1: return np.zeros(2)
        return diff / norm

    def step(self):
        dist_to_target = np.linalg.norm(self.pos - self.target)
        if dist_to_target < 2.0: return # Arrived
        
        step_size = 1.0
        
        if self.state == "SEEK":
            move_dir = self.get_gradient_dir()
            new_pos = self.pos + move_dir * step_size
            
            # Check collisions
            collided, obs_center = self.env.check_collision(new_pos)
            
            if collided:
                self.state = "WALL_FOLLOW"
                self.hit_point = self.pos.copy()
                self.dist_at_hit = dist_to_target
                # Decide turn direction (Left/Right) - Fixed Left for simplicity
            else:
                self.pos = new_pos
                
        elif self.state == "WALL_FOLLOW":
            # Wall Follow Logic: Move perpendicular to gradient of specific obstacle?
            # Or simplified: Circular motion around obstacle center (since circular obstacles)
            
            # Find nearest obstacle (cheat: returned by collision check)
            # Find tangent vector
            # Vec to center
            # Find current nearest obs
            min_dist = 1000
            nearest_obs = None
            for obs in self.env.obstacles:
                d = np.linalg.norm(self.pos - np.array([obs['x'], obs['y']]))
                if d < min_dist:
                    min_dist = d
                    nearest_obs = obs
                    
            if nearest_obs:
                # Tangent: (-y, x) of radius vector
                center = np.array([nearest_obs['x'], nearest_obs['y']])
                radius_vec = self.pos - center
                # Tangent (Counter-Clockwise)
                tangent = np.array([-radius_vec[1], radius_vec[0]])
                tangent = tangent / np.linalg.norm(tangent)
                
                new_pos = self.pos + tangent * step_size
                
                # Correction to stay on radius (Simple P controller)
                dist_err = np.linalg.norm(new_pos - center) - nearest_obs['r']
                correction = -0.5 * dist_err * (new_pos - center)/np.linalg.norm(new_pos - center)
                
                self.pos = new_pos + correction
                
                # BUG 2 Leave Condition:
                # 1. On M-Line? (Check cross product or angle)
                # 2. Closer than hit point?
                
                curr_dist_target = np.linalg.norm(self.pos - self.target)
                
                # Check M-line intersection (Approximate: Angle between Start-Target and Me-Target is 0 or 180)
                # Better: Distance from point to line < Threshold
                
                # Line: P_start -> P_target
                # Proj of vector P_start_Me onto P_start_Target
                # ...
                # Simple Logic for Lab: If angle to target matches original angle?
                
                # Let's use Bug 0 logic for simplicity + safety check
                # "Is the way to target clear?" (Raycast). 
                # If Clear, Leave. (Tangent Bug).
                
                # Let's simulate Tangent Bug
                move_dir = self.get_gradient_dir()
                test_pos = self.pos + move_dir * step_size * 2.0
                c, _ = self.env.check_collision(test_pos)
                
                # Hysteresis: Don't leave immediately
                if not c and curr_dist_target < self.dist_at_hit:
                     self.state = "SEEK"

def main():
    env = Environment()
    robots = []
    
    # Spawn 20 robots
    for i in range(20):
        start = [np.random.uniform(0, 20), np.random.uniform(0, 20)]
        robots.append(Robot(i, start, env.target, env))
        
    # Run
    history_x = []
    history_y = []
    
    print("Simulating Swarm...")
    for t in range(200): # 200 steps
        curr_x = []
        curr_y = []
        for r in robots:
            r.step()
            curr_x.append(r.pos[0])
            curr_y.append(r.pos[1])
        history_x.append(curr_x)
        history_y.append(curr_y)
        
    print("Plotting...")
    plt.figure(figsize=(10, 10))
    
    # Draw Obstacles
    for obs in env.obstacles:
        circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='black', alpha=0.5)
        plt.gca().add_patch(circle)
        
    # Draw Target
    plt.plot(env.target[0], env.target[1], 'y*', markersize=20, label='Light Source')
    
    # Draw Paths for a few robots
    hist_x = np.array(history_x)
    hist_y = np.array(history_y)
    
    for i in range(0, 20, 2): # Plot half of them
        plt.plot(hist_x[:, i], hist_y[:, i], '-', alpha=0.6)
        
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Swarm Gradient Search with Bug Algorithm")
    plt.legend()
    plt.grid()
    plt.savefig("output/swarm_viz.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Trap"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Robots hit the circular obstacles, slide around them, and peel off when the path is clear.
- **Modify:** Create a "C-Shape" obstacle (U-Trap) with the opening facing AWAY from the target.
- **Result:**
    *   **Gradient Descent:** Gets stuck in the bottom of the U. Local Minimum.
    *   **Bug 0:** Walks back and forth or gets stuck.
    *   **Bug 2:** Follows the *inside* of the U, then the *outside*, all the way around, until it exits the trap on the far side.
- **Lesson:** Gradient Descent fails in non-convex environments. Bug algos guarantee convergence (eventually).

---

## 🚀 Project: "Ant Colony Optimization (ACO)"

**Goal:** Find *Shortest* path cooperatively.
1.  **Exploration:** Robots move randomly.
2.  **Pheromones:** If Robot finds food, it returns home leaving a "Scent Trail".
3.  **Reinforcement:** Shorter paths get traversed more often $\to$ Scent gets stronger.
4.  **Decay:** Scent evaporates over time. Long paths fade out.
5.  **Result:** Swarm converges on shortest route.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Wall Follow Jitter"
*   **Cause:** Step size too large. Robot steps *inside* the obstacle, then pushes out, then steps in.
*   **Fix:** Smaller steps. Or Force interactions (Potential Fields repelling from wall).

#### 2. "Infinite Loop"
*   **Cause:** Bug 2 condition never met (Numerical precision on Line distance).
*   **Fix:** Use threshold $\epsilon$ for "Crossing M-Line".

---

## ⚡ Optimization: Potential Fields

Instead of Bug switching:
$$ F_{total} = F_{attract} + F_{repulse} $$
*   $F_{attract} \propto (Target - Pos)$.
*   $F_{repulse} \propto 1 / Dist_{wall}^2$.
*   **Issue:** Local Minima (Force Balance). Can use Random Walk or Simulated Annealing to escape.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Gradient vs Gradient Descent?
    *   **A:** Gradient points Uphill (Source Seeking). Gradient Descent points Downhill (Error Minimization). Swarm usually does Gradient Ascent (Seeking).
2.  **Q:** Computation requirement?
    *   **A:** Extremely low. Can run on 8-bit microcontrollers (Kilobots). No map storage needed.
3.  **Q:** Robustness?
    *   **A:** High. If 5 robots die, 15 still finish the job.

### Challenge Task
> **Task:** Light Intensity w/ Noise.
> 1. Add Gaussian noise to `get_signal`.
> 2. Robots will jitter.
> 3. Implement a Moving Average filter on the sensor limit.

---

## 📚 Further Reading
- **Lumelsky & Stepanov:** "Dynamic path planning for a mobile automaton with limited information on the environment" (The Bug Algo Paper).
- **Reynolds:** "Flocks, Herds, and Schools: A Distributed Behavioral Model" (Boids).

---

**Day 159 Complete**
