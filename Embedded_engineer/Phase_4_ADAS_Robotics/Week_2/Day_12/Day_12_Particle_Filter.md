# Day 12: The Particle Filter (Monte Carlo Localization)
## Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation

---

> **📝 Day 12 Focus:**
> Kalman Filters are great for tracking, but they assume the world is Gaussian and unimodal. What if the robot wakes up and has no idea where it is? Or what if it's in a symmetric hallway where it could be in two places at once? The **Particle Filter** solves this by representing the belief as a cloud of thousands of hypothetical robots.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** between Parametric (KF) and Non-Parametric (PF) filters.
2.  **Implement** the core steps of MCL: Initialization, Prediction, Weight Update, and Resampling.
3.  **Solve** the "Kidnapped Robot Problem" using global localization.
4.  **Develop** a Particle Filter in Python to localize a robot in a 2D landmark map.
5.  **Analyze** the trade-off between particle count, accuracy, and computational cost.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Probability:** Bayes Rule (Day 8).
-   **Python:** NumPy, Matplotlib.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `matplotlib`, `scipy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Non-Parametric Estimation

**Kalman Filter:**
-   Belief = Mean ($\mu$) + Covariance ($\Sigma$).
-   Assumption: Gaussian.
-   Limitation: Can only represent *one* hypothesis.

**Particle Filter:**
-   Belief = Set of $N$ particles $\{x^{(i)}, w^{(i)}\}$.
-   Assumption: None! Can represent *any* distribution.
-   Strength: Can represent multiple hypotheses (multimodal).

**The Intuition:**
Imagine 1000 ghost robots.
1.  **Move:** Move all ghosts according to the odometer (add noise).
2.  **Sense:** Compare what each ghost *would* see with what the real robot *actually* sees.
3.  **Weigh:** Ghosts that match the measurement get high weights. Ghosts that don't get low weights.
4.  **Resample:** Kill the low-weight ghosts. Clone the high-weight ghosts.
5.  **Repeat.**

### 🔹 Part 2: The MCL Algorithm

#### 2.1 Initialization
-   **Global Localization:** Scatter particles uniformly over the entire map.
-   **Tracking:** Scatter particles around a known start point (Gaussian).

#### 2.2 Prediction (Motion Update)
For each particle $i$:
$$ x_t^{(i)} = \text{motion\_model}(x_{t-1}^{(i)}, u_t) + \text{noise} $$

*Note: Noise is critical here. It spreads the particles out.*

#### 2.3 Update (Sensor Update)
For each particle $i$:
$$ w_t^{(i)} = \text{measurement\_model}(z_t, x_t^{(i)}) $$

Usually, the weight is the likelihood of the measurement given the particle's state.
$$ w \propto e^{-\frac{(z_{measured} - z_{expected})^2}{2\sigma^2}} $$

#### 2.4 Resampling (The Secret Sauce)
We create a new set of particles by drawing from the old set with probability proportional to their weights.
-   **Wheel of Fortune (Low Variance Sampling):** An efficient $O(N)$ algorithm to pick particles.
-   **Result:** Particles converge around the true state.

---

### 🔹 Part 3: Advanced Concepts

#### 3.1 The Kidnapped Robot Problem
If the robot is picked up and moved:
-   **KF:** Fails (diverges).
-   **Standard PF:** Fails (all particles are at the old location).
-   **Augmented PF:** Detects that *all* particle weights are low (poor match). Injects random particles across the map to re-localize.

#### 3.2 Number of Particles
-   Too few: Filter divergence (particle deprivation).
-   Too many: Slow computation.
-   **Adaptive PF (KLD-Sampling):** Adjusts $N$ dynamically based on the uncertainty (Kullback-Leibler Divergence).

---

## 💻 Implementation: 2D Particle Filter

We will simulate a robot moving in a 2D world with 4 landmarks.

### 🛠️ Setup
Create `week2_day12` and `particle_filter.py`.

```bash
mkdir -p ~/ros2_ws/src/week2_day12
cd ~/ros2_ws/src/week2_day12
touch particle_filter.py
```

### 👨‍💻 Code: Particle Filter Class

```python
import numpy as np
import matplotlib.pyplot as plt
import math

class ParticleFilter:
    def __init__(self, num_particles, map_size, landmarks):
        self.N = num_particles
        self.map_size = map_size # [width, height]
        self.landmarks = landmarks # [[x, y], ...]
        
        # Initialize particles uniformly
        self.particles = np.empty((self.N, 3)) # x, y, theta
        self.particles[:, 0] = np.random.uniform(0, map_size[0], self.N)
        self.particles[:, 1] = np.random.uniform(0, map_size[1], self.N)
        self.particles[:, 2] = np.random.uniform(0, 2*np.pi, self.N)
        
        self.weights = np.ones(self.N) / self.N

    def predict(self, u, std_pos, std_theta):
        # u = [velocity, yaw_rate]
        v, w = u
        dt = 0.1
        
        # Move each particle
        theta = self.particles[:, 2]
        
        if abs(w) > 0.001:
            self.particles[:, 0] += (v/w) * (np.sin(theta + w*dt) - np.sin(theta))
            self.particles[:, 1] += (v/w) * (np.cos(theta) - np.cos(theta + w*dt))
            self.particles[:, 2] += w*dt
        else:
            self.particles[:, 0] += v*dt * np.cos(theta)
            self.particles[:, 1] += v*dt * np.sin(theta)
            
        # Add Process Noise
        self.particles[:, 0] += np.random.normal(0, std_pos, self.N)
        self.particles[:, 1] += np.random.normal(0, std_pos, self.N)
        self.particles[:, 2] += np.random.normal(0, std_theta, self.N)
        
        # Normalize angles
        self.particles[:, 2] %= 2 * np.pi

    def update(self, z, std_landmark):
        # z = [[id, dist], ...]
        # Simple likelihood: Gaussian on distance
        
        self.weights.fill(1.0)
        
        for i in range(self.N):
            p_x, p_y = self.particles[i, 0], self.particles[i, 1]
            
            for measurement in z:
                lm_id = int(measurement[0])
                meas_dist = measurement[1]
                
                # Where the particle thinks the landmark is
                lm_x, lm_y = self.landmarks[lm_id]
                pred_dist = np.sqrt((p_x - lm_x)**2 + (p_y - lm_y)**2)
                
                # Gaussian Likelihood
                prob = (1.0 / (np.sqrt(2*np.pi) * std_landmark)) * \
                       np.exp(- (meas_dist - pred_dist)**2 / (2 * std_landmark**2))
                
                self.weights[i] *= prob
                
        # Normalize weights
        self.weights += 1.e-300 # Avoid division by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Low Variance Sampling (Wheel of Fortune)
        new_particles = np.empty_like(self.particles)
        
        beta = 0.0
        mw = np.max(self.weights)
        index = int(np.random.random() * self.N)
        
        for i in range(self.N):
            beta += np.random.random() * 2.0 * mw
            while beta > self.weights[index]:
                beta -= self.weights[index]
                index = (index + 1) % self.N
            new_particles[i] = self.particles[index]
            
        self.particles = new_particles

    def estimate(self):
        # Mean and Variance
        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var

def run_simulation():
    # Map
    landmarks = np.array([[10, 10], [10, 90], [90, 90], [90, 10]])
    map_size = [100, 100]
    
    pf = ParticleFilter(num_particles=1000, map_size=map_size, landmarks=landmarks)
    
    # Ground Truth Robot
    robot_pos = np.array([50.0, 50.0, 0.0])
    
    plt.figure(figsize=(10, 10))
    
    for t in range(50):
        # 1. Move Robot (Circle)
        u = [5.0, 0.1] # v=5, w=0.1
        dt = 0.1
        robot_pos[0] += u[0]*dt * np.cos(robot_pos[2])
        robot_pos[1] += u[0]*dt * np.sin(robot_pos[2])
        robot_pos[2] += u[1]*dt
        
        # 2. Simulate Measurements (Distance to all landmarks)
        z = []
        for i, lm in enumerate(landmarks):
            dist = np.sqrt((robot_pos[0]-lm[0])**2 + (robot_pos[1]-lm[1])**2)
            # Add noise
            dist += np.random.normal(0, 2.0)
            z.append([i, dist])
            
        # 3. PF Predict
        pf.predict(u, std_pos=0.5, std_theta=0.1)
        
        # 4. PF Update
        pf.update(z, std_landmark=2.0)
        
        # 5. PF Resample
        pf.resample()
        
        # Visualization
        plt.clf()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Particles
        plt.scatter(pf.particles[:, 0], pf.particles[:, 1], s=2, color='r', alpha=0.5, label='Particles')
        
        # Landmarks
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=100, marker='*', color='k', label='Landmarks')
        
        # Robot
        plt.plot(robot_pos[0], robot_pos[1], 'bo', markersize=10, label='Robot')
        
        # Estimate
        est_mean, est_var = pf.estimate()
        plt.plot(est_mean[0], est_mean[1], 'gx', markersize=10, label='Estimate')
        
        plt.legend()
        plt.title(f"Step {t}")
        plt.pause(0.1)

    plt.show()

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: Global Localization

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** Initially, particles are everywhere. After a few steps, they cluster around the robot.
3.  **Experiment:** Reduce `num_particles` to 10.
    -   *Result:* The filter will likely fail (cluster in the wrong place or die out).
4.  **Experiment:** Increase sensor noise (`std_landmark`).
    -   *Result:* The cluster will be larger (more uncertainty).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Particle Deprivation
**Symptom:** All particles are in the wrong place, and the robot is lost.
**Cause:** Resampling killed the good particles because of a temporary bad measurement or too few particles.
**Solution:**
-   Increase $N$.
-   Inject random particles (Augmented MCL).

#### 2. Sample Impoverishment
**Symptom:** All particles are identical (stacked on top of each other).
**Cause:** Process noise is too low. The particles don't spread out during prediction.
**Solution:** Increase `std_pos` and `std_theta` in the predict step.

#### 3. "Teleporting" Robot
**Symptom:** Estimate jumps wildly.
**Cause:** Multimodal distribution (e.g., symmetric hallway). The mean of two clusters (one at start, one at end) is in the middle (where the robot is NOT).
**Solution:** Don't use the mean. Use the "Best Particle" or cluster centroids.

---

## ⚡ Optimization & Best Practices

### 1. Vectorization
Python loops are slow. Vectorize the weight update:
```python
# Vectorized distance calculation
dx = particles[:, 0] - lm_x
dy = particles[:, 1] - lm_y
dist = np.hypot(dx, dy)
prob = np.exp(-(meas_dist - dist)**2 / (2*std**2))
weights *= prob
```

### 2. KLD Sampling (Adaptive)
Stop generating particles when the distribution is "dense enough".
-   If uncertainty is low, use 100 particles.
-   If uncertainty is high (global loc), use 10,000 particles.

### 3. AMCL (Adaptive Monte Carlo Localization)
This is the standard ROS 2 package (`nav2_amcl`). It implements KLD-sampling and Augmented MCL (random injection).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Can a Particle Filter handle non-linear models?
    *   **A:** Yes, naturally. You just pass the particle through the non-linear function. No Jacobians needed.
2.  **Q:** What is the computational complexity of PF?
    *   **A:** $O(N)$, where $N$ is the number of particles.
3.  **Q:** Why do we need resampling?
    *   **A:** Without it, the weights of all particles except one would eventually become effectively zero (degeneracy).

### Challenge Task
**Task:** Implement the "Kidnapped Robot" logic.
1.  Calculate the average weight of particles $w_{avg}$.
2.  If $w_{avg}$ drops below a threshold (meaning *no* particles explain the measurement well), replace 10% of particles with random ones.
3.  Teleport the robot in the simulation and watch it recover.

---

## 📚 Further Reading & References
-   [Probabilistic Robotics (Chapter 8)](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html) - The definitive guide to MCL.
-   [Nav2 AMCL Configuration](https://navigation.ros.org/configuration/packages/configuring-amcl.html)

---

**Day 12 Complete** | Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation
