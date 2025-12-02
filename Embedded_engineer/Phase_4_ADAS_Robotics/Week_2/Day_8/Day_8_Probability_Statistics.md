# Day 8: Probability & Statistics for Robotics
## Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation

---

> **📝 Day 8 Focus:**
> Robotics is the art of making decisions under uncertainty. Sensors are noisy, models are imperfect, and the world is unpredictable. To handle this, we use **Probability Theory**. Today, we lay the mathematical foundation for Sensor Fusion: Bayes' Rule, Gaussian Distributions, and Covariance Matrices.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** why robotics relies on probabilistic state estimation rather than deterministic logic.
2.  **Master** the properties of the Gaussian (Normal) Distribution (1D and Multivariate).
3.  **Apply** Bayes' Rule to update beliefs based on new sensor evidence.
4.  **Interpret** Covariance Matrices and their geometric representation (Error Ellipses).
5.  **Implement** a Python simulation of a discrete Bayes Filter.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Linear Algebra:** Matrix multiplication, Determinants, Inversion.
-   **Calculus:** Basic integrals (area under curve).
-   **Python:** NumPy, Matplotlib.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `scipy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Uncertainty in Robotics

#### 1.1 Sources of Uncertainty
1.  **Sensor Noise:** LiDAR points jitter, Cameras have grain, GPS drifts.
2.  **Actuation Noise:** You tell the car to move 1.0m, it moves 1.05m (wheel slip).
3.  **Model Errors:** The map says the wall is flat, but it's curved.
4.  **Algorithmic Approximations:** We linearize non-linear functions.

#### 1.2 Belief State
Instead of saying "The car is at x=5", we say:
"The car is *probably* around x=5, with a standard deviation of 0.5m."

Mathematically, the state $x$ is a **Random Variable**, and our knowledge is a **Probability Density Function (PDF)**, $p(x)$.

---

### 🔹 Part 2: The Gaussian Distribution

The "Bell Curve" is the workhorse of robotics (Kalman Filters).

#### 2.1 Univariate (1D) Gaussian
Defined by Mean ($\mu$) and Variance ($\sigma^2$):

$$ p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{ -\frac{(x-\mu)^2}{2\sigma^2} } $$

-   **Mean ($\mu$):** The center (expected value).
-   **Variance ($\sigma^2$):** The spread (uncertainty).
-   **Standard Deviation ($\sigma$):** $\sqrt{\text{Variance}}$.

**Properties:**
-   68% of data falls within $\mu \pm 1\sigma$.
-   95% of data falls within $\mu \pm 2\sigma$.
-   99.7% of data falls within $\mu \pm 3\sigma$.

#### 2.2 Multivariate (ND) Gaussian
For a state vector $\mathbf{x}$ (e.g., $[x, y, \theta]$):

$$ p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^k |\mathbf{\Sigma}|}} e^{ -\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) } $$

-   **Mean Vector ($\mathbf{\mu}$):** Center of the blob.
-   **Covariance Matrix ($\mathbf{\Sigma}$):** Describes the shape and orientation of the blob.

**The Covariance Matrix:**
For 2D state $[x, y]$:
$$ \mathbf{\Sigma} = \begin{bmatrix} \sigma_x^2 & \sigma_{xy} \\ \sigma_{yx} & \sigma_y^2 \end{bmatrix} $$

-   $\sigma_x^2$: Variance in x.
-   $\sigma_y^2$: Variance in y.
-   $\sigma_{xy}$: Correlation between x and y.
    -   Positive: If x is high, y is likely high.
    -   Zero: x and y are independent (Circle/Axis-aligned ellipse).

---

### 🔹 Part 3: Bayes' Rule

This is the update mechanism. How do we combine "What we thought before" (Prior) with "What we just saw" (Likelihood)?

$$ p(x | z) = \frac{p(z | x) \cdot p(x)}{p(z)} $$

-   **$p(x)$ - Prior:** Belief *before* measurement. (Prediction).
-   **$p(z | x)$ - Likelihood:** Probability of seeing measurement $z$ *given* state $x$. (Sensor Model).
-   **$p(x | z)$ - Posterior:** Belief *after* measurement. (Correction).
-   **$p(z)$ - Normalizer:** Ensures probabilities sum to 1.

**In Robotics terms:**
$$ \text{Posterior} \propto \text{Likelihood} \times \text{Prior} $$

**Example:**
1.  **Prior:** GPS says I am at location 10 ($\pm 5$).
2.  **Measurement:** Camera sees a landmark known to be at location 12.
3.  **Posterior:** I am likely between 10 and 12, and my uncertainty is *lower* than both GPS and Camera alone.

---

## 💻 Implementation: Python Simulation

We will implement:
1.  **Gaussian Visualizer:** Plotting 1D and 2D Gaussians.
2.  **Discrete Bayes Filter:** Simulating a robot in a hallway.

### 🛠️ Setup
Create a folder `week2_day8` and a file `bayes_sim.py`.

```bash
mkdir -p ~/ros2_ws/src/week2_day8
cd ~/ros2_ws/src/week2_day8
touch bayes_sim.py
```

### 👨‍💻 Code: Gaussian Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_1d_gaussian(mu, sigma):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(- (x - mu)**2 / (2 * sigma**2))
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f'N({mu}, {sigma}^2)')
    plt.fill_between(x, y, alpha=0.2)
    plt.title("1D Gaussian Distribution")
    plt.xlabel("State (x)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid()
    plt.show()

def plot_2d_gaussian(mu, sigma_matrix):
    x, y = np.mgrid[mu[0]-3:mu[0]+3:.01, mu[1]-3:mu[1]+3:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mu, sigma_matrix)
    
    plt.figure(figsize=(6, 6))
    plt.contourf(x, y, rv.pdf(pos), levels=20, cmap='viridis')
    plt.title(f"2D Gaussian\nCov: {sigma_matrix}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.colorbar(label='Probability Density')
    plt.show()

# Test
if __name__ == "__main__":
    # 1D
    plot_1d_gaussian(mu=0, sigma=1.0)
    
    # 2D Uncorrelated
    plot_2d_gaussian(mu=[0, 0], sigma_matrix=[[1, 0], [0, 1]])
    
    # 2D Correlated (Stretched diagonal)
    plot_2d_gaussian(mu=[0, 0], sigma_matrix=[[1, 0.8], [0.8, 1]])
```

### 👨‍💻 Code: Discrete Bayes Filter (Grid Localization)

Simulate a robot moving in a 1D hallway with 20 cells.
-   **Map:** `[0, 0, 1, 0, 0]` (1 = Door, 0 = Wall).
-   **Sensor:** Can detect if it's in front of a door (with noise).
-   **Motion:** Moves 1 cell right (with noise).

```python
def discrete_bayes_filter():
    # --- Configuration ---
    world_map = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0] # 1=Door
    n_cells = len(world_map)
    
    # Initial Belief (Uniform - we don't know where we are)
    p = np.ones(n_cells) / n_cells
    
    # Noise Models
    p_hit = 0.8  # Correct sensor reading
    p_miss = 0.2 # Incorrect sensor reading
    
    p_exact = 0.8 # Move exactly 1 step
    p_overshoot = 0.1 # Move 2 steps
    p_undershoot = 0.1 # Move 0 steps
    
    # --- Functions ---
    
    def sense(p, z):
        """
        Update belief p based on measurement z (1=Door, 0=Wall)
        """
        q = np.zeros_like(p)
        for i in range(len(p)):
            hit = (z == world_map[i])
            # Likelihood
            q[i] = p[i] * (p_hit if hit else p_miss)
            
        # Normalize
        q = q / np.sum(q)
        return q

    def move(p, u):
        """
        Predict belief p based on motion u (steps)
        Convolution of belief and motion kernel
        """
        q = np.zeros_like(p)
        for i in range(len(p)):
            # Probability of coming from i-u (exact)
            # We use modulo for cyclic world (or clamp for bounded)
            # Here assuming cyclic for simplicity
            
            # Contribution from exact move
            src = (i - u) % len(p)
            q[i] += p[src] * p_exact
            
            # Contribution from overshoot
            src = (i - u - 1) % len(p)
            q[i] += p[src] * p_overshoot
            
            # Contribution from undershoot
            src = (i - u + 1) % len(p)
            q[i] += p[src] * p_undershoot
            
        return q

    # --- Simulation Loop ---
    
    # Real robot state
    true_pos = 0
    
    plt.figure(figsize=(12, 6))
    
    for t in range(10):
        # 1. Move Robot
        true_pos = (true_pos + 1) % n_cells
        
        # 2. Predict (Motion Update)
        p = move(p, 1)
        
        # 3. Sense
        # Generate noisy measurement
        is_door = world_map[true_pos]
        if np.random.rand() < p_hit:
            z = is_door
        else:
            z = 1 - is_door # Sensor error
            
        # 4. Correct (Measurement Update)
        p = sense(p, z)
        
        # Visualization
        plt.clf()
        plt.bar(range(n_cells), p, alpha=0.6, color='b', label='Belief')
        plt.axvline(true_pos, color='r', linestyle='--', label='True Pos')
        
        # Draw Map (Doors)
        for i, val in enumerate(world_map):
            if val == 1:
                plt.plot(i, -0.05, 'ks', markersize=10) # Black square for door
                
        plt.title(f"Step {t}: True Pos={true_pos}, Meas={'Door' if z else 'Wall'}")
        plt.ylim(-0.1, 1.0)
        plt.legend()
        plt.pause(1.0)

    plt.show()

if __name__ == "__main__":
    discrete_bayes_filter()
```

---

## 🔬 Lab Exercise: Covariance Tuning

### Lab Objectives
1.  Run the Gaussian visualizer.
2.  Modify the covariance matrix to see how the ellipse changes.
3.  Answer: What does a diagonal covariance matrix imply?

### Experiments
1.  **Identity:** `[[1, 0], [0, 1]]` -> Circle. X and Y are independent.
2.  **Scaled:** `[[5, 0], [0, 1]]` -> Wide ellipse. High uncertainty in X, low in Y.
3.  **Correlated:** `[[1, 0.9], [0.9, 1]]` -> Thin diagonal line. If X is known, Y is known.
4.  **Anti-Correlated:** `[[1, -0.9], [-0.9, 1]]` -> Thin diagonal line (negative slope).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Probabilities don't sum to 1
**Cause:** Forgetting to normalize after the measurement update.
**Solution:** `p = p / np.sum(p)`

#### 2. Matrix not Positive Semi-Definite
**Symptom:** `multivariate_normal` throws error.
**Cause:** Invalid covariance matrix (e.g., correlation > 1).
**Solution:** Ensure diagonal elements are positive, and determinant is positive.

#### 3. "Kidnapped Robot Problem"
**Symptom:** Belief converges to wrong location and gets stuck.
**Cause:** Probability becomes 0 at the true location due to sensor noise/mismatch. Once 0, it can never become non-zero (multiplication by 0).
**Solution:** Add a small non-zero "floor" probability to all cells (random particle injection).

---

## ⚡ Optimization & Best Practices

### 1. Log-Likelihood
In implementation, multiplying many small probabilities leads to **underflow** (floating point becomes 0).
**Solution:** Work in Log-space.
$$ \log(p(x|z)) = \log(p(z|x)) + \log(p(x)) - \log(p(z)) $$
Multiplication becomes Addition.

### 2. Mahalanobis Distance
How far is a point $x$ from the mean $\mu$, considering the covariance $\Sigma$?
$$ D_M(x) = \sqrt{ (x-\mu)^T \Sigma^{-1} (x-\mu) } $$
This is the "Standard Deviation" equivalent for multivariate distributions. Used for **Gating** (rejecting outliers).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** If the Covariance Matrix is diagonal, what does it mean?
    *   **A:** The variables are uncorrelated (independent).
2.  **Q:** What happens to the uncertainty (variance) after a measurement update?
    *   **A:** It decreases (usually). Information gain reduces entropy.
3.  **Q:** What happens to the uncertainty after a motion update (prediction)?
    *   **A:** It increases. Motion adds noise.

### Challenge Task
**Task:** Implement a 2D Histogram Filter.
1.  Create a 10x10 grid.
2.  Robot moves in X and Y.
3.  Sensor returns (x, y) with noise.
4.  Visualize the 2D heatmap of belief.

---

## 📚 Further Reading & References
-   [Probabilistic Robotics (Thrun, Burgard, Fox)](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html) - The Bible of Robotics.
-   [Kalman Filter Visualization](https://www.kalmanfilter.net/)

---

**Day 8 Complete** | Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation
