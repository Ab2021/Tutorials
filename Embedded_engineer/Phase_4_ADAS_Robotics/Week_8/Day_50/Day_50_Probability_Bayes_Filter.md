# Day 50: Probability & Bayes Filter
## Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion

---

> **📝 Day 50 Focus:**
> Robots are never 100% sure. "I think I'm at x=5, but I might be at x=5.1." Uncertainty is inherent in sensors and motion. Today, we embrace this uncertainty using **Probability Theory** and the **Bayes Filter**, the mathematical foundation of all modern robotics.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the difference between Frequentist and Bayesian probability.
2.  **Apply** Bayes' Rule to update beliefs based on new measurements.
3.  **Implement** the Bayes Filter algorithm: **Predict** (Motion) and **Update** (Measurement).
4.  **Code** a Discrete Bayes Filter (Histogram Filter) for 1D localization.
5.  **Visualize** how the probability distribution evolves over time.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Math:** Basic Algebra.
-   **Python:** Lists, Loops.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Robot's Belief

We represent the robot's state $x$ (position) not as a single number, but as a **Probability Distribution** $p(x)$.
-   **Belief:** $bel(x_t) = p(x_t | z_{1:t}, u_{1:t})$.
    -   Probability of being at $x$ at time $t$, given all past measurements $z$ and controls $u$.

### 🔹 Part 2: Bayes' Rule

$$ P(x | z) = \frac{P(z | x) P(x)}{P(z)} $$

-   **$P(x)$ - Prior:** What we thought before seeing the sensor data.
-   **$P(z | x)$ - Likelihood:** How likely is this measurement $z$ if we were at $x$? (Sensor Model).
-   **$P(x | z)$ - Posterior:** What we think now.
-   **$P(z)$ - Normalizer:** Ensures probabilities sum to 1.

### 🔹 Part 3: The Bayes Filter Algorithm

Recursive loop running at every time step:

1.  **Prediction (Motion Update):**
    -   Move the distribution according to control $u$.
    -   Add uncertainty (Motion Noise).
    -   $\overline{bel}(x_t) = \int p(x_t | u_t, x_{t-1}) bel(x_{t-1}) dx$

2.  **Correction (Measurement Update):**
    -   Sharpen the distribution using sensor data $z$.
    -   $bel(x_t) = \eta P(z_t | x_t) \overline{bel}(x_t)$

---

## 💻 Implementation: 1D Localization

**Scenario:**
-   A robot moves in a 1D hallway with 100 cells.
-   **Map:** The hallway has 3 doors at indices [20, 50, 80].
-   **Sensor:** Detects "Door" or "Wall". Accuracy: 80%.
-   **Motion:** Moves 1 cell forward. Accuracy: 90% (10% undershoot/overshoot).

### 🛠️ Setup
Create `week8_day50` and `bayes_filter.py`.

```bash
mkdir -p ~/ros2_ws/src/week8_day50
cd ~/ros2_ws/src/week8_day50
touch bayes_filter.py
```

### 👨‍💻 Code: Discrete Bayes Filter

```python
import numpy as np
import matplotlib.pyplot as plt
import time

class BayesFilter1D:
    def __init__(self, size, doors):
        self.size = size
        self.doors = doors
        # Initial Belief: Uniform (Lost)
        self.belief = np.ones(size) / size
        
        # Sensor Model
        self.p_hit = 0.8 # Correct detection
        self.p_miss = 0.2 # False detection
        
        # Motion Model
        self.p_exact = 0.8
        self.p_overshoot = 0.1
        self.p_undershoot = 0.1

    def predict(self, u):
        # u: distance to move (e.g., 1)
        new_belief = np.zeros(self.size)
        
        for i in range(self.size):
            # Convolution
            if self.belief[i] > 0:
                # Exact move
                idx = (i + u) % self.size
                new_belief[idx] += self.p_exact * self.belief[i]
                
                # Overshoot
                idx = (i + u + 1) % self.size
                new_belief[idx] += self.p_overshoot * self.belief[i]
                
                # Undershoot
                idx = (i + u - 1) % self.size
                new_belief[idx] += self.p_undershoot * self.belief[i]
                
        self.belief = new_belief

    def update(self, z):
        # z: measurement ("door" or "wall")
        for i in range(self.size):
            # Is there actually a door here?
            is_door = (i in self.doors)
            
            # Likelihood P(z|x)
            likelihood = 1.0
            if z == "door":
                likelihood = self.p_hit if is_door else self.p_miss
            else: # z == "wall"
                likelihood = self.p_hit if not is_door else self.p_miss
                
            self.belief[i] *= likelihood
            
        # Normalize
        self.belief /= np.sum(self.belief)

def run_simulation():
    size = 100
    doors = [20, 50, 80]
    bf = BayesFilter1D(size, doors)
    
    # Ground Truth
    robot_pos = 0
    
    plt.figure(figsize=(10, 5))
    
    for t in range(100):
        # 1. Move Robot (Ground Truth)
        robot_pos = (robot_pos + 1) % size
        
        # 2. Predict (Filter)
        bf.predict(1)
        
        # 3. Sense (Ground Truth)
        # Simulate noisy sensor
        is_door_gt = (robot_pos in doors)
        if np.random.rand() < 0.8:
            measurement = "door" if is_door_gt else "wall"
        else:
            measurement = "wall" if is_door_gt else "door" # Noise
            
        # 4. Update (Filter)
        bf.update(measurement)
        
        # 5. Visualization
        plt.clf()
        plt.bar(range(size), bf.belief, color='b', label='Belief')
        plt.axvline(robot_pos, color='r', linestyle='--', label='True Pos')
        
        # Draw Doors
        for d in doors:
            plt.axvline(d, color='k', linewidth=5, alpha=0.3, ymin=0, ymax=0.1)
            
        plt.title(f"Step {t} | Measurement: {measurement}")
        plt.legend()
        plt.ylim(0, 1.0)
        plt.pause(0.1)

    plt.show()

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: The "Kidnapped Robot"

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** Initially, the belief is flat (Uniform).
3.  **Event:** As the robot passes the first door (Index 20), the belief spikes at 20, 50, and 80 (Multimodal). It knows it's at *a* door, but not *which* one.
4.  **Event:** As it moves to the second door, the belief collapses to a single peak. The distance between doors (30 vs 40) disambiguates the location.
5.  **Kidnapping:**
    -   Modify the code to teleport `robot_pos` to 0 at step 50.
    -   *Result:* The filter will be confused (Belief -> 0). It takes time to recover. This is the **Kidnapped Robot Problem**.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Belief becomes Zero
**Symptom:** `NaN` or all zeros.
**Cause:** Measurement was "impossible" according to the model (Likelihood = 0), or float underflow.
**Solution:** Never use 0.0 probability. Use a small epsilon (0.0001).

#### 2. Lag
**Symptom:** The peak is behind the true position.
**Cause:** Motion model underestimates the movement, or update rate is too slow.
**Solution:** Tune `p_exact` vs `p_undershoot`.

---

## ⚡ Optimization & Best Practices

### 1. Log-Odds
Multiplying many small probabilities causes underflow ($0.1^{100} \approx 0$).
**Log-Odds:** Work with logarithms.
-   $log(P(x|z)) = log(P(z|x)) + log(P(x)) - log(P(z))$.
-   Multiplication becomes Addition. Much more stable.

### 2. Continuous vs Discrete
Grid filters are great for 1D/2D.
For 3D (6-DOF), grids are too expensive ($N^6$).
We need **Kalman Filters** (Gaussian assumption) or **Particle Filters** (Sampling).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Markov Assumption"?
    *   **A:** The future depends only on the current state, not the history. $P(x_t | x_{t-1}, x_{t-2}...) = P(x_t | x_{t-1})$.
2.  **Q:** Why do we need the "Predict" step?
    *   **A:** Sensors only tell us where we *are*. Motion tells us where we *went*. Without prediction, we ignore the robot's movement.
3.  **Q:** What happens if the sensor is perfect ($P=1.0$)?
    *   **A:** The belief becomes a delta function (100% certainty).

### Challenge Task
**Task:** 2D Grid Localization.
1.  Create a 10x10 grid.
2.  Map: A 2D array with obstacles.
3.  Motion: Up, Down, Left, Right.
4.  Sensor: Returns "Obstacle Nearby" (True/False).
5.  Implement the 2D Bayes Filter.

---

## 📚 Further Reading & References
-   [Probabilistic Robotics (Thrun, Burgard, Fox)](http://www.probabilistic-robotics.org/) - The Bible.
-   [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

---

**Day 50 Complete** | Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion
