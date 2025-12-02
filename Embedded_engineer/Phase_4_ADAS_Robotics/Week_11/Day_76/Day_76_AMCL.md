# Day 76: Robot Localization (AMCL)
## Phase 4: ADAS & Robotics Systems | Week 11: Localization

---

> **📝 Day 76 Focus:**
> You have a map. You have a LiDAR. You have Odometry. But where exactly are you on the map? **AMCL (Adaptive Monte Carlo Localization)** is the industry standard for indoor localization. It uses thousands of "particles" (guesses) to figure out the robot's pose.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Particle Filter algorithm (MCL) applied to Localization.
2.  **Implement** the Motion Model (Odometry) and Sensor Model (LiDAR Beam).
3.  **Code** a complete AMCL system in Python.
4.  **Solve** the "Kidnapped Robot Problem" using random particle injection.
5.  **Tune** AMCL parameters (Particle count, Resampling threshold).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 52:** Particle Filter Basics.
-   **Day 74:** Odometry.
-   **Probability:** Gaussian Distribution.

### Hardware Requirements
-   **LiDAR:** (Optional) 2D Laser Scanner.

### Software Stack
-   **Python:** `numpy`, `matplotlib`, `scipy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Particle Filter (Recap)

We represent the posterior belief $p(x_t | z_{1:t}, u_{1:t})$ as a set of $M$ weighted samples (particles).
$$ S_t = \{ \langle x_t^{[m]}, w_t^{[m]} \rangle \}_{m=1}^M $$

### 🔹 Part 2: Motion Model (Prediction)

For each particle $m$:
$$ x_t^{[m]} = \text{sample\_motion\_model}(u_t, x_{t-1}^{[m]}) $$
-   Add noise to the odometry control $u_t$.
-   Propagate the particle.
-   **Result:** The cloud spreads out (uncertainty increases).

### 🔹 Part 3: Sensor Model (Correction)

For each particle $m$:
$$ w_t^{[m]} = \text{measurement\_model}(z_t, x_t^{[m]}, \text{map}) $$
-   Simulate a LiDAR scan from the particle's pose.
-   Compare simulated scan with real scan.
-   **Likelihood Field:** Pre-compute a distance map from obstacles.
    $$ P(z | x, m) \propto \exp \left( - \frac{\text{dist}^2}{2\sigma^2} \right) $$

### 🔹 Part 4: Adaptive Resampling (KLD)

-   **Standard MCL:** Fixed number of particles (e.g., 1000).
-   **Adaptive (KLD-Sampling):**
    -   If uncertainty is low (particles clustered), use fewer particles (e.g., 100).
    -   If uncertainty is high (particles spread), use more (e.g., 5000).
    -   Saves CPU.

---

## 💻 Implementation: AMCL from Scratch

**Scenario:**
-   **Map:** A simple 2D room with landmarks.
-   **Sensor:** Range-only sensor (simplified LiDAR).
-   **Goal:** Localize the robot.

### 🛠️ Setup
Create `week11_day76` and `amcl.py`.

```bash
mkdir -p ~/ros2_ws/src/week11_day76
cd ~/ros2_ws/src/week11_day76
touch amcl.py
```

### 👨‍💻 Code: AMCL Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Constants ---
WORLD_SIZE = 100.0
LANDMARKS = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]]
N_PARTICLES = 1000

class Robot:
    def __init__(self):
        self.x = np.random.random() * WORLD_SIZE
        self.y = np.random.random() * WORLD_SIZE
        self.orientation = np.random.random() * 2.0 * np.pi
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        
    def set(self, new_x, new_y, new_orientation):
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation) % (2.0 * np.pi)
        
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        self.forward_noise = float(new_f_noise)
        self.turn_noise = float(new_t_noise)
        self.sense_noise = float(new_s_noise)
        
    def sense(self):
        z = []
        for i in range(len(LANDMARKS)):
            dist = math.sqrt((self.x - LANDMARKS[i][0])**2 + (self.y - LANDMARKS[i][1])**2)
            dist += np.random.normal(0.0, self.sense_noise)
            z.append(dist)
        return z
    
    def move(self, turn, forward):
        if forward < 0:
            raise ValueError('Robot cant move backwards')
            
        # Add noise to command
        orientation = self.orientation + float(turn) + np.random.normal(0.0, self.turn_noise)
        orientation %= 2 * np.pi
        
        dist = float(forward) + np.random.normal(0.0, self.forward_noise)
        x = self.x + (math.cos(orientation) * dist)
        y = self.y + (math.sin(orientation) * dist)
        x %= WORLD_SIZE    # cyclic world
        y %= WORLD_SIZE
        
        # Create new robot with updated state
        res = Robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res
    
    def measurement_prob(self, measurement):
        prob = 1.0
        for i in range(len(LANDMARKS)):
            dist = math.sqrt((self.x - LANDMARKS[i][0])**2 + (self.y - LANDMARKS[i][1])**2)
            # Gaussian
            prob *= self.gaussian(dist, self.sense_noise, measurement[i])
        return prob
    
    def gaussian(self, mu, sigma, x):
        return math.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / math.sqrt(2.0 * math.pi * (sigma ** 2))

def main():
    # 1. Initialize Robot
    myrobot = Robot()
    myrobot.set_noise(5.0, 0.1, 5.0)
    myrobot.set(30.0, 50.0, np.pi/2)
    
    # 2. Initialize Particles (Uniform Distribution)
    p = []
    for i in range(N_PARTICLES):
        r = Robot()
        r.set_noise(5.0, 0.1, 5.0)
        p.append(r)
        
    # Simulation Loop
    steps = 20
    
    plt.figure(figsize=(8, 8))
    
    for t in range(steps):
        # Move Robot
        myrobot = myrobot.move(0.1, 5.0)
        z = myrobot.sense()
        
        # Move Particles
        p2 = []
        for i in range(N_PARTICLES):
            p2.append(p[i].move(0.1, 5.0))
        p = p2
        
        # Weight Particles
        w = []
        for i in range(N_PARTICLES):
            w.append(p[i].measurement_prob(z))
            
        # Resample (Wheel)
        p3 = []
        index = int(np.random.random() * N_PARTICLES)
        beta = 0.0
        mw = max(w)
        for i in range(N_PARTICLES):
            beta += np.random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % N_PARTICLES
            p3.append(p[index])
        p = p3
        
        # Visualization
        plt.cla()
        plt.xlim(0, WORLD_SIZE)
        plt.ylim(0, WORLD_SIZE)
        
        # Draw Landmarks
        for lm in LANDMARKS:
            plt.plot(lm[0], lm[1], 'ks', markersize=10)
            
        # Draw Particles
        px = [part.x for part in p]
        py = [part.y for part in p]
        plt.plot(px, py, 'g.', alpha=0.3, label='Particles')
        
        # Draw Robot
        plt.plot(myrobot.x, myrobot.y, 'ro', markersize=10, label='Robot')
        # Arrow
        plt.arrow(myrobot.x, myrobot.y, 5*math.cos(myrobot.orientation), 5*math.sin(myrobot.orientation), head_width=2, color='r')
        
        plt.title(f"Step {t}")
        plt.legend()
        plt.pause(0.2)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Kidnapped Robot

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** Particles converge to the red robot quickly.
3.  **Experiment:**
    -   At step 10, teleport the robot: `myrobot.set(80.0, 80.0, 0.0)`.
    -   **Result:** The particles stay at the old location. The robot is "lost".
    -   **Fix:** Add **Random Particle Injection**.
    -   Logic: `if avg_weight < threshold: add_random_particles()`.
    -   This allows the filter to "re-discover" the robot at the new location.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Particle Depletion
**Symptom:** All particles cluster in a wrong spot.
**Cause:** Sensor noise `sigma` is too small. The filter is "too confident" and rejects the true pose because it's slightly off.
**Solution:** Increase `sense_noise`. It makes the likelihood function wider/flatter.

#### 2. Lag
**Symptom:** Particles trail behind the robot.
**Cause:** Motion model noise is too small. The particles don't spread enough to cover the actual movement.
**Solution:** Increase `forward_noise` and `turn_noise`.

---

## ⚡ Optimization & Best Practices

### 1. Likelihood Field
Calculating distance to nearest obstacle for every ray for every particle is slow.
-   **Pre-compute** a Distance Transform (Likelihood Field) of the map.
-   Lookup is $O(1)$.

### 2. KLD Sampling
Don't use 1000 particles if 50 will do.
-   Calculate the Kullback-Leibler Divergence (distance between particle distribution and true distribution).
-   Stop generating particles when KLD is small enough.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Resampling" step?
    *   **A:** Selecting particles with high weights and duplicating them, while discarding low-weight particles. It focuses computation on likely areas.
2.  **Q:** Why is AMCL better than EKF for global localization?
    *   **A:** EKF assumes a Gaussian (unimodal) distribution. It can't handle "I might be in Room A OR Room B". Particle filters are multi-modal.
3.  **Q:** What happens if the map is wrong?
    *   **A:** AMCL fails. It assumes the map is the ground truth.

### Challenge Task
**Task:** Ray Casting Sensor Model.
1.  Instead of Landmarks, use a grid map (occupancy grid).
2.  Implement `ray_cast(x, y, theta, map)` to find distance to wall.
3.  Compare with `measurement`.

---

## 📚 Further Reading & References
-   [Probabilistic Robotics (Thrun, Burgard, Fox)](https://docs.ros.org/en/nav2/concepts/index.html)
-   [Nav2 AMCL Configuration](https://navigation.ros.org/configuration/packages/configuring-amcl.html)

---

**Day 76 Complete** | Phase 4: ADAS & Robotics Systems | Week 11: Localization
