# Day 52: Particle Filter (Monte Carlo Localization)
## Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion

---

> **📝 Day 52 Focus:**
> The Kalman Filter assumes the world is Gaussian (Bell Curve). But what if the robot is in a symmetrical corridor? It could be at $x=10$ OR $x=50$. A Gaussian can't represent "OR". The **Particle Filter** can. It represents belief as a cloud of thousands of hypothetical robots.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** Parametric Filters (KF/EKF) with Non-Parametric Filters (PF).
2.  **Explain** the MCL Algorithm: Resampling, Prediction, Weight Update.
3.  **Implement** the "Low Variance Resampling" wheel.
4.  **Code** a Particle Filter in Python to localize a robot in a 2D map with landmarks.
5.  **Solve** the "Kidnapped Robot Problem" (Global Localization).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 50:** Bayes Filter.
-   **Probability:** Sampling from distributions.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Particle Representation

Instead of Mean $\mu$ and Covariance $\Sigma$, we have a set of $N$ particles:
$$ \mathcal{X}_t = \{ \langle x_t^{[1]}, w_t^{[1]} \rangle, \dots, \langle x_t^{[N]}, w_t^{[N]} \rangle \} $$
-   $x^{[i]}$: A hypothesis of the state (e.g., "I am at x=5, y=3").
-   $w^{[i]}$: The weight (probability) of that hypothesis.

### 🔹 Part 2: The MCL Algorithm

1.  **Resampling:**
    -   Draw $N$ particles from the previous set $\mathcal{X}_{t-1}$ with probability proportional to their weights $w$.
    -   Result: High-weight particles are duplicated; low-weight particles die out.
2.  **Prediction (Motion):**
    -   Move *each* particle according to the motion model + random noise.
    -   $x_t^{[i]} = f(x_{t-1}^{[i]}, u_t) + \epsilon$.
3.  **Update (Measurement):**
    -   Calculate the weight of each particle based on how well it matches the sensor data.
    -   $w_t^{[i]} = P(z_t | x_t^{[i]})$.

### 🔹 Part 3: Resampling Wheel

Naive resampling ($O(N \log N)$) is slow.
**Low Variance Resampling ($O(N)$):**
-   Imagine a roulette wheel where slice size = weight.
-   Instead of spinning $N$ times, we spin *once* and have $N$ equally spaced pointers.
-   Ensures diversity and efficiency.

---

## 💻 Implementation: 2D MCL

**Scenario:**
-   **Map:** 4 Landmarks at corners of a 50x50m area.
-   **Robot:** Moves in a circle.
-   **Sensor:** Measures distance to landmarks (Range-only).

### 🛠️ Setup
Create `week8_day52` and `particle_filter.py`.

```bash
mkdir -p ~/ros2_ws/src/week8_day52
cd ~/ros2_ws/src/week8_day52
touch particle_filter.py
```

### 👨‍💻 Code: Particle Filter Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
DT = 0.1
SIM_TIME = 50.0
NUM_PARTICLES = 100
FIELD_SIZE = 60.0

# Landmarks [x, y]
LANDMARKS = np.array([
    [10.0, 10.0],
    [50.0, 10.0],
    [50.0, 50.0],
    [10.0, 50.0]
])

# Noise Parameters
Q_motion = np.diag([0.1, np.deg2rad(10.0)]) ** 2  # [v, omega]
R_meas = 3.0 ** 2  # Range measurement variance

class ParticleFilter:
    def __init__(self, num_particles):
        self.np = num_particles
        # Initialize particles uniformly
        self.particles = np.random.rand(self.np, 3) * FIELD_SIZE
        self.particles[:, 2] = np.random.rand(self.np) * 2 * np.pi - np.pi
        self.weights = np.ones(self.np) / self.np

    def predict(self, u):
        # Move each particle
        v = u[0]
        w = u[1]
        
        # Add noise to control input
        v_noisy = v + np.random.randn(self.np) * np.sqrt(Q_motion[0, 0])
        w_noisy = w + np.random.randn(self.np) * np.sqrt(Q_motion[1, 1])
        
        # Motion Model (Differential Drive)
        self.particles[:, 0] += v_noisy * np.cos(self.particles[:, 2]) * DT
        self.particles[:, 1] += v_noisy * np.sin(self.particles[:, 2]) * DT
        self.particles[:, 2] += w_noisy * DT

    def update(self, z):
        # z: List of ranges to landmarks
        for i in range(self.np):
            w = 1.0
            x, y, theta = self.particles[i]
            
            for j in range(len(LANDMARKS)):
                # Expected distance
                dx = x - LANDMARKS[j, 0]
                dy = y - LANDMARKS[j, 1]
                dist_pred = math.sqrt(dx**2 + dy**2)
                
                # Measured distance
                dist_meas = z[j]
                
                # Gaussian Likelihood
                # P(z|x) = exp(-(z - z_pred)^2 / (2 * R))
                prob = math.exp(-((dist_meas - dist_pred)**2) / (2 * R_meas))
                w *= prob
            
            self.weights[i] = w
            
        # Normalize weights
        self.weights += 1.e-300 # Avoid zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Low Variance Resampling
        new_particles = []
        step = 1.0 / self.np
        r = np.random.rand() * step
        c = self.weights[0]
        i = 0
        
        for m in range(self.np):
            U = r + m * step
            while U > c:
                i = (i + 1) % self.np
                c += self.weights[i]
            new_particles.append(self.particles[i])
            
        self.particles = np.array(new_particles)
        self.weights = np.ones(self.np) / self.np
        
    def estimate(self):
        # Weighted Mean
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        # Angle average is tricky, use vector sum
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        theta = math.atan2(sin_sum, cos_sum)
        return np.array([x, y, theta])

def main():
    pf = ParticleFilter(NUM_PARTICLES)
    
    # Ground Truth
    xTrue = np.array([30.0, 30.0, 0.0])
    
    time = 0.0
    while time <= SIM_TIME:
        time += DT
        
        # 1. Control
        u = np.array([1.0, 0.1])
        
        # 2. Move Robot (Ground Truth)
        xTrue[0] += u[0] * math.cos(xTrue[2]) * DT
        xTrue[1] += u[0] * math.sin(xTrue[2]) * DT
        xTrue[2] += u[1] * DT
        
        # 3. Measure (Simulate Range Sensor)
        z = []
        for lm in LANDMARKS:
            dist = math.sqrt((xTrue[0] - lm[0])**2 + (xTrue[1] - lm[1])**2)
            dist += np.random.randn() * math.sqrt(R_meas) # Add noise
            z.append(dist)
            
        # 4. PF Steps
        pf.predict(u)
        pf.update(z)
        xEst = pf.estimate()
        pf.resample()
        
        # 5. Visualization
        if int(time * 10) % 5 == 0:
            plt.cla()
            plt.xlim(0, FIELD_SIZE)
            plt.ylim(0, FIELD_SIZE)
            
            # Plot Landmarks
            plt.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "ok", markersize=10)
            
            # Plot Particles
            plt.plot(pf.particles[:, 0], pf.particles[:, 1], ".r", alpha=0.5)
            
            # Plot True vs Est
            plt.plot(xTrue[0], xTrue[1], "xb", markersize=10, label="True")
            plt.plot(xEst[0], xEst[1], "+g", markersize=10, label="Est")
            
            plt.legend()
            plt.pause(0.001)
            
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Global Localization

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Initially, red dots (particles) are everywhere.
    -   After a few steps, they cluster around the blue cross (True Position).
    -   This is **Global Localization**. The robot started "Lost" and found itself.
3.  **Comparison:**
    -   Try doing this with an EKF. You can't. EKF needs an initial guess ($x_0$). If you give it the wrong guess, it diverges. PF handles the ambiguity naturally.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Particle Deprivation
**Symptom:** All particles cluster in a tiny spot, but it's the wrong spot.
**Cause:** Resampling too aggressively or not enough noise in prediction.
**Solution:** Add "Jitter" (random noise) after resampling. Or inject random particles (Kidnapped Robot recovery).

#### 2. Weights -> 0
**Symptom:** Division by zero error.
**Cause:** Sensor measurement is extremely unlikely (far from all particles).
**Solution:** Check sensor model variance ($R$). If $R$ is too small, the Gaussian is too narrow. Widen it.

---

## ⚡ Optimization & Best Practices

### 1. Adaptive MCL (AMCL)
Fixed $N$ is inefficient.
-   When lost: Need many particles (10,000).
-   When localized: Need few particles (100).
-   **KLD-Sampling:** Adjust $N$ dynamically based on the "tightness" (Kullback-Leibler Divergence) of the belief.

### 2. Ray Casting
Calculating `dist_pred` for Lidar (360 rays) for 1000 particles is slow ($3.6 \times 10^5$ ray casts per step).
-   **Likelihood Field:** Pre-compute a "Distance Map" of the world.
-   Look up the distance in the map instead of ray casting. $O(1)$.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is Resampling necessary?
    *   **A:** Without resampling, the weights of all particles except one would eventually become effectively zero (Particle Degeneracy). We need to focus computational power on the likely areas.
2.  **Q:** What is the "Kidnapped Robot Problem"?
    *   **A:** The robot is picked up and moved to a new location without being told (no odometry). It must realize its sensor data doesn't match its belief and re-localize globally.
3.  **Q:** Can PF handle the "Symmetric Hallway" problem?
    *   **A:** Yes. The particles will split into two clusters (multi-modal belief), one at each possible location, until a unique landmark is seen.

### Challenge Task
**Task:** Kidnapping Recovery.
1.  Modify `resample()` to add 5% random particles every step.
2.  Teleport the robot (`xTrue`) to a new corner at $t=25$.
3.  Observe: The main cluster dies out (low weights). The random particles near the new location get high weights and spawn a new cluster.

---

## 📚 Further Reading & References
-   [Monte Carlo Localization (Thrun et al.)](https://www.cs.cmu.edu/~16831-f14/notes/F11/16831_lecture04_mcl.pdf)
-   [ROS 2 AMCL Package](https://github.com/ros-planning/navigation2/tree/main/nav2_amcl)

---

**Day 52 Complete** | Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion
