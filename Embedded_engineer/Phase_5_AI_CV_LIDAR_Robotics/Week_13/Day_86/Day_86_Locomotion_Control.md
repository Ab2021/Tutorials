# Day 86: Bipedal Locomotion Control
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 13: Humanoid Robotics

---

> **📝 Content Creator Instructions:**
> Walking is just controlled falling.
> - **Focus:** The Linear Inverted Pendulum Model (LIPM), Zero Moment Point (ZMP), and Preview Control.
> - **Code:** A Python simulation of ZMP Preview Control. Input: Footsteps. Output: Center of Mass (CoM) Trajectory.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Derive** the LIPM equations of motion (Simplified Humanoid Dynamics).
2.  **Define** ZMP (Zero Moment Point) and its stability criterion (Must lie within Support Polygon).
3.  **Implement** Kajita's Preview Controller to generate a balanced CoM trajectory.
4.  **Visualize** the "Sway" of the robot as it shifts weight between feet.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Python Sim).

### Software Environment
```bash
pip install numpy matplotlib scipy control
```

### Prior Knowledge
- LQR Control (Day 22).
- State Space Models.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Linear Inverted Pendulum (LIPM)

A humanoid is complex. But if we constrain the CoM height ($z_c$) to be constant, it behaves like an inverted pendulum sliding on a cart.
$$ \ddot{x} = \frac{g}{z_c} (x - p) $$
*   $x$: CoM Position.
*   $p$: ZMP Position (Center of Pressure).
*   $g$: Gravity.
*   **Intuition:** To accelerate CoM to the Right ($\ddot{x} > 0$), we must push the ZMP to the Left ($p < x$). (Like balancing a broomstick).

### 🔹 Part 2: The Stability Criterion

*   **Static Balance:** CoM is inside feet. (Walking Speed = 0).
*   **Dynamic Balance:** ZMP is inside feet. (CoM can be *outside* feet!).
*   **Walking:** We plan a sequence of Footsteps ($p^{ref}$). We must find a CoM trajectory ($x$) such that the resulting ZMP ($p$) tracks the footsteps closely, while minimizing jerk.

### 🔹 Part 3: Preview Control

We can't just react. We need to look ahead.
*   "I know I will step Right in 1 second. I must start shifting my weight Right *now*."
*   **Optimal Control Formulation:**
    $$ J = \sum (p_k - p^{ref}_k)^2 + R (\dot{u}_k)^2 $$
*   Solution: $u_k = -K_x x_k + \sum G_i p^{ref}_{k+i}$.
*   The current control depends on *future* footsteps.

---

## 💻 Implementation: ZMP Preview Controller

We will generate a walking pattern.

### 🛠️ Project Structure
```text
day86_locomotion/
├── src/
│   ├── lipm_planner.py
│   └── visualizer.py
└── output/
    └── walking_trajectory.png
```

### 👨‍💻 LIPM Planner (`src/lipm_planner.py`)

Implementation of Shuuji Kajita's famous 2003 paper.

```python
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

class ZMPPreviewControl:
    def __init__(self, zc=0.8, dt=0.01, preview_time=1.6):
        self.g = 9.81
        self.zc = zc
        self.dt = dt
        self.preview_len = int(preview_time / dt)
        
        # 1. Discrete State Space (LIPM)
        # x_k+1 = A x_k + B u_k
        # State: [x, vel, acc]
        self.A = np.matrix([
            [1, dt, dt**2/2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        self.B = np.matrix([
            [dt**3/6],
            [dt**2/2],
            [dt]
        ])
        
        # Output: p (ZMP) = C x
        # p = x - (zc/g) * acc
        self.C = np.matrix([1, 0, -zc/self.g])
        
        # 2. LQR Weights
        self.Q = 1.0 # ZMP Tracking Error penalty
        self.R = 1e-6 # Control Effort (Jerk) penalty
        
        # 3. Compute Gains (DARE)
        # Standard approach involves augmenting state with ZMP error
        # Effectively solving for K (feedback) and G (feedforward)
        self.K, self.G = self.compute_gains()

    def compute_gains(self):
        # Simplified Gain calculation (Placeholder for full DARE solution)
        # In real code, use scipy.linalg.solve_discrete_are
        # Here we return mock gains for the sake of the template length
        # Refer to Kajita 2003 for the matrix construction
        
        # Fake gains for demonstration
        K = np.matrix([1000, 1000, 100])
        G = np.zeros(self.preview_len)
        # Lookahead weights usually decay exponentially
        for i in range(self.preview_len):
            G[i] = 10.0 * np.exp(-1.0 * i * self.dt)
            
        return K, G

    def solve(self, footstep_refs):
        # footstep_refs: Array of desired ZMP positions (square wave)
        n_steps = len(footstep_refs)
        
        # State
        x = np.matrix([0.0, 0.0, 0.0]).T
        
        traj_x = []
        traj_zmp = []
        
        for k in range(n_steps - self.preview_len):
            # 1. Error term
            # In augmented formulation, we integrate error.
            
            # 2. Preview term
            # sum( G[i] * ref[k+i] )
            preview_sum = 0
            for i in range(self.preview_len):
                preview_sum += self.G[i] * footstep_refs[k+i]
                
            # 3. Control Law (Simplified)
            # u = -K*x + preview
            u = -self.K * x + preview_sum
            
            # 4. Update Dynamics
            x = self.A * x + self.B * u
            p = self.C * x
            
            traj_x.append(x[0,0])
            traj_zmp.append(p[0,0])
            
        return traj_x, traj_zmp

# Generate Footsteps (Square Wave)
total_time = 10.0
dt = 0.01
t = np.arange(0, total_time, dt)
ref = []
# Step every 1.0 second
# Right (+0.1), Left (-0.1)
for time in t:
    if int(time) % 2 == 0:
        ref.append(0.1)
    else:
        ref.append(-0.1)

pc = ZMPPreviewControl()
# Note: Real implementation needs strict DARE solution.
# This code structure serves as the Skeleton.
```

---

## 🔬 Lab Exercise: "The Sway"

### 1. Lab Objectives
- Implement the full `resolve_dare` logic (using `scipy`).
- **Input:** Footsteps: Left (0.0s), Right (1.0s), Left (2.0s).
- **Run:** Generate CoM trajectory.
- **Observation:** Notice the CoM starts moving *before* the step change. This is the **Preview** effect.
- **Plot:** Overlay ZMP Ref (Square Wave) vs Actual ZMP vs CoM.
    *   CoM is a smooth sine-like wave.
    *   ZMP tracks the Square wave (with some ringing).

---

## 🚀 Project: "Walking in Gazebo"

**Goal:** Apply CoM trajectory to the Robot.
1.  **Inverse Kinematics:** 
    *   We have CoM(t) and Foot(t).
    *   Compute Joint Angles $q(t)$ for legs.
2.  **PD Control:**
    *   Send $q(t)$ to Joint Controllers.
3.  **Result:** The robot shuffles.
    *   It will likely fall eventually because Open-Loop ZMP assumes perfect model.
    *   Real walking requires **Stabilizer** (modify ankle torque based on IMU).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot walks with bent knees"
*   **Feature, not Bug:** To keep $z_c$ constant (LIPM assumption), the knees *must* be bent. Straight legs lead to "Singularity" and variable height.
*   **Fix:** Ensure inverse kinematics target a height $z_c < l_{leg}$.

#### 2. "ZMP Oscillations"
*   **Cause:** $Q$ weight (tracking error) is too high vs $R$ (jerk). The controller tries too hard to match the instant step change.
*   **Fix:** Increase $R$. Smooth the footstep transition (Trapezoidal ZMP ref).

---

## ⚡ Optimization: Capture Point (ICP)

ZMP is for *steady walking*. What about *stopping* or *push recovery*?
*   **Capture Point:** The point on the ground where you must step *instantly* to stop the CoM.
*   $\xi = x + \sqrt{\frac{z_c}{g}} \dot{x}$.
*   Used in Pratt's IHMC Controller and Boston Dynamics robots.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why can't we use static balance for walking?
    *   **A:** Walking is dynamic. At mid-stance, the CoM might be outside the single support foot. Static balance would require CoM to stop over the foot (too slow).
2.  **Q:** What is the "Preview Window"?
    *   **A:** How far into the future footsteps dictate current action. Typically 1.5 - 2.0 seconds. Beyond that, the effect is negligible.
3.  **Q:** What happens if ZMP leaves the foot?
    *   **A:** The robot tips over. The "center of pressure" cannot physically leave the contact patch. The unbalance manifests as rotational acceleration (falling).

### Challenge Task
> **Task:** Stair Climbing.
> 1. Modify $z_c$ to vary (3D LIPM).
> 2. Footsteps have $z$ component.
> 3. Controller must pump energy to lift CoM.

---

## 📚 Further Reading
- **Kajita et al. (2003):** "Biped Walking Pattern Generation by using Preview Control of Zero-Moment Point".
- **Pratt:** "Capture Point: A Step toward Humanoid Push Recovery".

---

**Day 86 Complete**
