# Day 128: Particle Filter (Monte Carlo Localization)
## Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM

---

> **📝 Day 128 Focus:**
> Kalman Filters assume everything is Gaussian (Bell Curve). But what if the robot could be in Room A *or* Room B? This is a **Multi-Modal** distribution. The **Particle Filter (PF)** handles this by simulating thousands of possible robot poses.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the core steps: Predict, Update, Resample.
2.  **Compare** Particle Filter vs. Kalman Filter.
3.  **Implement** a 1D Particle Filter from scratch.
4.  **Solve** the "Kidnapped Robot Problem" (Global Localization).
5.  **Visualize** particle convergence.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Probability:** Probability Density Functions (PDF).
-   **Day 114:** Bayes Filter logic.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `scipy.stats`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Concept

Instead of tracking Mean ($\mu$) and Variance ($\Sigma$), we track $N$ **Particles**.
Each particle is a hypothesis: "I think the robot is at $x=5$".
-   **Prediction:** Move all particles according to motion model + noise.
-   **Update:** Weigh particles based on sensor match. (High match = High weight).
-   **Resample:** Kill low-weight particles, duplicate high-weight particles.

### 🔹 Part 2: The Algorithm (MCL)

1.  **Initialization:** Scatter particles uniformly (Global Localization) or around a guess (Tracking).
2.  **Motion Update:** $x_i = x_i + u + \text{noise}$.
3.  **Sensor Update:** $w_i = P(z | x_i)$. (Likelihood).
4.  **Resampling:** Draw $N$ new particles with probability proportional to $w_i$.

### 🔹 Part 3: The Kidnapped Robot

If you pick up a robot and move it, the KF fails (it thinks it's still at the old spot).
The PF can recover *if* we inject random particles occasionally. If the sensor readings don't match the main cloud, the random particles might match the new location and grow.

---

## 💻 Implementation: 1D Particle Filter

**Scenario:**
-   Robot in a hallway with 3 doors.
-   Sensor: Detects "Door" or "Wall".
-   Task: Find position.

### 🛠️ Setup
Create `week19_day128` and `particle_filter.py`.

```bash
mkdir -p ~/ros2_ws/src/week19_day128
cd ~/ros2_ws/src/week19_day128
touch particle_filter.py
```

### 👨‍💻 Code: MCL Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class ParticleFilter:
    def __init__(self, num_particles, map_size, landmarks):
        self.N = num_particles
        self.map_size = map_size
        self.landmarks = landmarks # Positions of doors
        
        # Initialize Uniformly
        self.particles = np.random.uniform(0, map_size, self.N)
        self.weights = np.ones(self.N) / self.N

    def predict(self, u, std_u):
        # Move particles
        self.particles += u + np.random.normal(0, std_u, self.N)
        # Wrap around (Circular world for simplicity)
        self.particles %= self.map_size

    def update(self, z, std_z):
        # z: Distance to nearest door (Measurement)
        # For each particle, calculate expected measurement
        
        for i in range(self.N):
            # Find closest landmark to this particle
            dist = np.min(np.abs(self.particles[i] - self.landmarks))
            
            # Likelihood: Gaussian(dist, std_z).pdf(z)
            # We measure 'z', expected is 'dist'.
            prob = norm(dist, std_z).pdf(z)
            self.weights[i] = prob
            
        # Normalize weights
        self.weights += 1.e-300 # Avoid div by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Systematic Resampling
        indices = np.random.choice(self.N, self.N, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        mean = np.mean(self.particles)
        var = np.var(self.particles)
        return mean, var

def main():
    # Setup
    map_size = 100.0
    doors = np.array([20.0, 50.0, 80.0]) # Doors at 20, 50, 80
    
    pf = ParticleFilter(num_particles=1000, map_size=map_size, landmarks=doors)
    
    # Robot State
    robot_pos = 10.0
    
    plt.figure(figsize=(10, 6))
    
    for t in range(20):
        # 1. Move Robot
        move = 2.0
        robot_pos += move
        robot_pos %= map_size
        
        # 2. Measure
        # True distance to nearest door
        true_dist = np.min(np.abs(robot_pos - doors))
        z = true_dist + np.random.normal(0, 1.0) # Noisy measurement
        
        # 3. PF Cycle
        pf.predict(u=move, std_u=1.0)
        pf.update(z, std_z=1.0)
        
        # Plot before resampling to see weights
        if t % 5 == 0:
            plt.subplot(2, 2, int(t/5)+1)
            plt.hist(pf.particles, bins=50, weights=pf.weights, range=(0, 100), density=True, alpha=0.7, color='b')
            plt.axvline(robot_pos, color='r', linestyle='--', label='Robot')
            for d in doors:
                plt.axvline(d, color='g', linewidth=2, label='Door' if d==doors[0] else "")
            plt.title(f"Step {t}")
            if t == 0: plt.legend()
            
        pf.resample()
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Ambiguity

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   At $t=0$, particles are everywhere (Uniform).
    -   At $t=5$, particles cluster around *all three* doors. The robot sees a door, but doesn't know *which* door. (Multi-modal).
    -   As the robot moves, the clusters shift. Eventually, only one cluster survives because the sequence of measurements (Door... Wall... Wall... Door) matches only one location in the map.
3.  **Experiment:**
    -   Reduce `num_particles` to 10.
    -   **Result:** Filter divergence. Not enough particles to cover the state space. This is **Particle Deprivation**.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Particle Deprivation
**Symptom:** All particles congregate at the wrong spot.
**Cause:** Resampling kills diversity too fast.
**Solution:** Add random particles (Injection) or use Low Variance Resampling.

#### 2. Computational Cost
**Symptom:** Slow update.
**Cause:** Too many particles ($N=100,000$).
**Solution:** Adaptive MCL (KLD-Sampling). Adjust $N$ dynamically. If uncertainty is low, use fewer particles.

---

## ⚡ Optimization & Best Practices

### 1. Low Variance Resampling
Standard `random.choice` is noisy.
-   **Wheel Algorithm:** Imagine a roulette wheel. Spin it once, then take steps of size $1/N$.
-   Ensures we pick particles proportionally without random fluctuation.
-   $O(N)$ complexity instead of $O(N \log N)$.

### 2. Sensor Model
The likelihood $P(z|x)$ is critical.
-   **Beam Model:** Ray casting. Accurate but slow.
-   **Likelihood Field:** Pre-compute a distance map. Fast lookup.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is PF better than KF for global localization?
    *   **A:** KF assumes a unimodal Gaussian. It cannot represent "I am at Door A OR Door B". It would average them and say "I am in the wall between A and B".
2.  **Q:** What is the "Curse of Dimensionality"?
    *   **A:** Number of particles needed grows exponentially with dimensions. 1D needs 100. 3D needs 10,000. 6D needs 1,000,000.
3.  **Q:** What happens during Resampling?
    *   **A:** High-weight particles are duplicated. Low-weight particles are deleted. The population evolves to track the high-probability regions.

### Challenge Task
**Task:** Kidnapping.
1.  At $t=15$, force `robot_pos = 80.0` (Teleport).
2.  Observe the filter fail (it tracks the old position).
3.  Modify `resample`: Replace 5% of particles with random uniform particles.
4.  Observe recovery.

---

## 📚 Further Reading & References
-   [Probabilistic Robotics (Thrun) - Chapter 4](https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf)
-   [MCL Visualization](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb)

---

**Day 128 Complete** | Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM
