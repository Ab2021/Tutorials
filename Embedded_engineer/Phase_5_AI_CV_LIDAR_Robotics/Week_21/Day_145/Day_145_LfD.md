# Day 145: Learning from Demonstration (Kinesthetic Teaching)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 21: Collaborative Robotics (Cobots)

---

> **📝 Content Creator Instructions:**
> Show, don't tell.
> - **Focus:** Programming by Demonstration (PbD), Kinesthetic Teaching (Gravity Comp Mode), Dynamic Movement Primitives (DMP) vs Waypoints, and Generalizing skills.
> - **Code:** A Python tool `record_replay.py` that captures Mouse movements (Simulated Teaching), encodes them into a path, and replays them with "Generalization" (Start/End point modification).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Use** Kinesthetic Teaching (free-drive) to record robotic tasks without writing code.
2.  **Compare** Replaying Raw Joint Angles vs Replaying Cartesian Paths.
3.  **Explain** Dynamic Movement Primitives (DMP): How to stretch/warp a recorded motion to a new target $X_{new}$.
4.  **Implement** a simple "Record and Rescale" pipeline.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Mouse (as proxy for Robot Handle). Or PyBullet with Drag interactions.

### Software Environment
```bash
pip install numpy matplotlib scipy
```

### Prior Knowledge
- Trajectory Interpolation.
- Differential Equations.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: No Code Robotics

Cobots (UR, Franka) dominate because you don't need Python to program them. You grab the arm, move it, and say "Save".
*   **Gravity Compensation:** Essential. Robot must feel weightless.
*   **Data Recording:** Encoders record $(q_1, q_2... q_6)$ at 100Hz.

### 🔹 Part 2: The Problem with Raw Replay

If you record "Pick cup at A, Place at B", and then move the cup to A', raw replay fails (Robot grasps air at A).
*   **Solution:** Parameterized Trajectories.
*   **DMP (Dynamic Movement Primitives):** Encodes the "Shape" of the motion as a forcing function in a spring-damper system.
    $$ \tau \dot{v} = K (g - x) - D v + (g - x_0) f(s) $$
    *   $g$: Goal.
    *   $f(s)$: Learned shape profile.
    *   Changing $g$ stretches the trajectory while keeping the shape.

### 🔹 Part 3: Gaussian Mixture Models (GMM)

If you demonstrate the task 5 times, how does the robot average them?
*   **GMM:** Finds the "Mean Path" and the "Variance" (Tube).
*   **Stiffness Control:** Where variance is low (Precision insert), be stiff. Where variance is high (Transfer motion), act compliant.

---

## 💻 Implementation: Mouse Teaching

We record mouse trajectory and replay it scaled.

### 🛠️ Project Structure
```text
day145_lfd/
├── src/
│   ├── record_replay.py
└── output/
    ├── generalization.png
```

### 👨‍💻 Record & Replay (`src/record_replay.py`)

Using `matplotlib` interactive events to capture "Teacher" input.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class TrajectoryLearner:
    def __init__(self):
        self.recorded_path = []
        self.is_recording = False
        
    def on_press(self, event):
        if event.button == 1: # Left Click
            self.is_recording = True
            self.recorded_path = []
            print("Recording started...")

    def on_release(self, event):
        if event.button == 1:
            self.is_recording = False
            print(f"Recording stopped. {len(self.recorded_path)} points.")
            self.plot_result()

    def on_move(self, event):
        if self.is_recording and event.inaxes:
            self.recorded_path.append([event.xdata, event.ydata])

    def generalize(self, new_start, new_goal):
        # 1. Normalize recorded path (0 to 1)
        path = np.array(self.recorded_path)
        if len(path) < 2: return None
        
        # Original Start/Goal
        p0 = path[0]
        pg = path[-1]
        
        # Vector from p0 to pg
        vec_orig = pg - p0
        len_orig = np.linalg.norm(vec_orig)
        angle_orig = np.arctan2(vec_orig[1], vec_orig[0])
        
        # New Vector
        vec_new = new_goal - new_start
        len_new = np.linalg.norm(vec_new)
        angle_new = np.arctan2(vec_new[1], vec_new[0])
        
        # Rotation Matrix (Align new vector with old vector)
        # Or better: Transform each point relative to p0 into the new frame.
        
        # Simple Approach: Affine Transform
        # Scaling Factor
        scale = len_new / (len_orig + 1e-6)
        rotation = angle_new - angle_orig
        
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        
        new_path = []
        for p in path:
            # Shift to origin
            p_local = p - p0
            # Scale & Rotate
            p_transformed = (R @ p_local) * scale
            # Shift to new start
            p_final = p_transformed + new_start
            new_path.append(p_final)
            
        return np.array(new_path)

    def plot_result(self):
        plt.close() # Close interactive
        
        path = np.array(self.recorded_path)
        
        # Generate varied targets
        gen1 = self.generalize(np.array([0,0]), np.array([5,5]))
        gen2 = self.generalize(np.array([0,0]), np.array([5, -2]))
        
        plt.figure(figsize=(10,6))
        
        # Original
        if len(path) > 0:
            plt.plot(path[:,0], path[:,1], 'k--', linewidth=2, label='Teacher (Demo)')
            plt.plot(path[0,0], path[0,1], 'go')
            plt.plot(path[-1,0], path[-1,1], 'rx')
        
        # Generalized
        if gen1 is not None:
            plt.plot(gen1[:,0], gen1[:,1], 'b-', label='Replay (Target A)')
            plt.plot(gen1[-1,0], gen1[-1,1], 'bx')
            
        if gen2 is not None:
            plt.plot(gen2[:,0], gen2[:,1], 'm-', label='Replay (Target B)')
            plt.plot(gen2[-1,0], gen2[-1,1], 'mx')
            
        plt.legend()
        plt.title("Learning from Demonstration (Affine Warp)")
        plt.grid()
        plt.axis('equal')
        plt.savefig("output/generalization.png")
        print("Plot saved. Run again to record new path.")
        plt.show()

def main():
    print("INSTRUCTIONS: Click and Drag on the plot to draw a path.")
    learner = TrajectoryLearner()
    
    fig, ax = plt.subplots()
    ax.set_title("Draw here")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    fig.canvas.mpl_connect('button_press_event', learner.on_press)
    fig.canvas.mpl_connect('button_release_event', learner.on_release)
    fig.canvas.mpl_connect('motion_notify_event', learner.on_move)
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Obstacle"

### 1. Lab Objectives
- **Run:** Draw a "U" shape (as if avoiding an obstacle in the middle).
- **Observe:** The Generalize function preserves the "U" shape even when reaching a target far away.
- **Fail:** If the new Start/Goal are very close, the "U" becomes tiny (Scaling).
- **Critique:** Affine Warp is dumb. It scales *width* with *length*.
- **Task:** Try DMP (Conceptually). DMP separates the "Forcing Term" (Shape) from the "Canonical System" (Time). It preserves the "Height" of the U even if the distance changes.

---

## 🚀 Project: "Task Segmentation"

**Goal:** Teach a complex task "Pick -> Pour -> Place".
1.  **Record:** One long continuous motion.
2.  **Algorithm:** Detect zero-velocity points.
    *   Stop 1: Grasp.
    *   Stop 2: Pour Start.
    *   Stop 3: Pour End.
    *   Stop 4: Release.
3.  **Result:** Break recording into 3 primitives: `Pick`, `Pour`, `Place`.
4.  **Replay:** Execute them sequentially.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Jittery Replay"
*   **Cause:** Human hands shake. Recording rate (mouse move events) is uneven.
*   **Fix:** Smoothing (Savitzky-Golay filter) or B-Spline Interpolation before encoding.

#### 2. "Overfitting"
*   **Cause:** Recording only 1 demo.
*   **Result:** Robot copies mistakes.
*   **Fix:** Record 5 demos. Average them.

---

## ⚡ Optimization: Inverse Reinforcement Learning (IRL)

Instead of copying the *Trajectory*, infer the *Cost Function*.
*   **Teacher:** Avoids the table edge carefully.
*   **Student (IRL):** "Ah, distance to table edge has high penalty."
*   **Result:** Generalizes to new tables better than trajectory warping.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Gravity Compensation?
    *   **A:** The controller calculates torques required to hold the arm against gravity and applies them. The user only feels friction/inertia.
2.  **Q:** Why not just use Inverse Kinematics (IK)?
    *   **A:** IK finds *final pose*. LfD teaches the *path* (e.g., to avoid an obstacle or pour liquid without spilling).
3.  **Q:** DMP vs Neural Net?
    *   **A:** DMP guarantees stability (Spring-damper). Neural Nets (Behavior Cloning) can drift or oscillate if OOD (Out of Distribution).

### Challenge Task
> **Task:** Speed Control.
> 1. In `generalize`, adjust the `dt` during playback.
> 2. `dt_new = dt_old * 0.5` (2x Speed).
> 3. Calculate Velocities. Check if they exceed robot limits.

---

## 📚 Further Reading
- **Ijspeert et al.:** "Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors".
- **Billard et al.:** "Robot Programming by Demonstration".

---

**Day 145 Complete**
