# Day 55: Visual Servoing (Eye-in-Hand)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 8: Advanced Manipulation

---

> **📝 Content Creator Instructions:**
> We don't need calibration. We just need to zero the error in the image.
> - **Focus:** Image-Based Visual Servoing (IBVS), Interaction Matrix (Image Jacobian), and Eye-in-Hand vs Eye-to-Hand.
> - **Code:** Controlling a camera to "lock on" to an AprilTag and track it.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** IBVS (Control in pixels) vs PBVS (Control in 3D Pose).
2.  **Derive** the Image Jacobian relating Camera Velocity $v_c$ to Feature Velocity $\dot{s}$.
3.  **Implement** a servo loop: $\dot{e} = -\lambda (s - s_{des}) \to v_c = -\lambda J_{img}^{+} e$.
4.  **Perform** a peg-alignment task using only a wrist camera.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Camera mounted on Robot End Effector.
- AprilTag.

### Software Environment
```bash
pip install opencv-python numpy apriltag
```

### Prior Knowledge
- Pinhole Camera Model (Day 9).
- Pseudo-Inverse (Day 51).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Eye-in-Hand vs Eye-to-Hand

1.  **Eye-in-Hand:** Camera on wrist. Moves with robot.
    *   *Pros:* Resolution increases as you approach target. No occlusion by robot body.
    *   *Cons:* FOV changes. Motion blur.
2.  **Eye-to-Hand:** Camera fixed on table.
    *   *Pros:* Global view. Stable image.
    *   *Cons:* Occlusion. Lower accuracy grasp.

### 🔹 Part 2: Image-Based Visual Servoing (IBVS)

Control error is defined in the **Image Plane** ($u, v$).
*   Target: Center of image ($u=320, v=240$).
*   Current: Feature location ($u=200, v=240$).
*   Error: $e = -120$ pixels.
*   Action: Move camera "Right".

### 🔹 Part 3: The Interaction Matrix $L_s$ (Image Jacobian)

How does a 3D velocity $(v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)$ affect a pixel $(u, v)$?
$$ \dot{s} = L_s(u, v, Z) \cdot V_c $$
$$ L_s = \begin{bmatrix}
-1/Z & 0 & u/Z & uv & -(1+u^2) & v \\
0 & -1/Z & v/Z & 1+v^2 & -uv & -u
\end{bmatrix} $$
*   **Key:** Depends on Depth $Z$. We must estimate $Z$ (Lidar/Stereo) or approximate it (assume constant).

---

## 💻 Implementation: IBVS Tracker

We will simulate a camera moving to center a Red Dot.
Simplified 2-DOF case ($v_x, v_y$).

### 🛠️ Project Structure
```text
day55_servoing/
├── src/
│   ├── image_jacobian.py
│   ├── controller.py
│   └── visual_sim.py
└── run_ibvs.py
```

### 👨‍💻 Image Processing (`src/visual_sim.py`)

Simulates a Virtual Camera viewing a point at $(X, Y, Z)$ in camera frame.

```python
import numpy as np

class VirtualCamera:
    def __init__(self):
        # Intrinsics
        self.fx = 800
        self.fy = 800
        self.cx = 320
        self.cy = 240
        self.width = 640
        self.height = 480
        
        # Target Point in World Frame (Fixed)
        self.P_world = np.array([0.5, 0.5, 2.0])
        
        # Camera Pose in World (Initial)
        self.Cam_pos = np.array([0.0, 0.0, 0.0])
        
    def get_image_feature(self):
        # Transform World Point to Camera Frame
        # P_cam = P_world - Cam_pos
        P_c = self.P_world - self.Cam_pos
        
        # Project to Image Plane
        Z = P_c[2]
        if Z <= 0.1: return None, Z # Behind camera
        
        u = (P_c[0] / Z) * self.fx + self.cx
        v = (P_c[1] / Z) * self.fy + self.cy
        
        # Normalize coordinates (x = (u-cx)/fx)
        x_n = (u - self.cx) / self.fx
        y_n = (v - self.cy) / self.fy
        
        return np.array([x_n, y_n]), Z
        
    def move(self, v_cam, dt=0.01):
        # Update Camera Position
        self.Cam_pos += v_cam * dt
```

### 👨‍💻 Controller (`src/controller.py`)

$$ v_c = -\lambda L_s^{+} e $$

```python
import numpy as np

def compute_velocity(feature, Z_est, desired=[0,0]):
    # Feature: [x, y] normalized coords
    x, y = feature
    des_x, des_y = desired
    
    # Error
    e = np.array([x - des_x, y - des_y])
    
    # Interaction Matrix for Pure Translation (vx, vy, vz)
    # L_s = [[-1/Z, 0, x/Z], [0, -1/Z, y/Z]]
    # We ignore rotation for this snippet
    
    L = np.array([
        [-1.0/Z_est, 0.0, x/Z_est],
        [0.0, -1.0/Z_est, y/Z_est]
    ])
    
    # Pseudo-Inverse
    L_inv = np.linalg.pinv(L)
    
    # Control Law: v = -lambda * L_inv * e
    lam = 5.0 # Gain
    v_cmd = -lam * np.dot(L_inv, e)
    
    return v_cmd, e
```

### 👨‍💻 Main Loop (`run_ibvs.py`)

```python
import matplotlib.pyplot as plt
from src.visual_sim import VirtualCamera
from src.controller import compute_velocity
import numpy as np

cam = VirtualCamera()
history_u = []
history_v = []

print("Starting IBVS...")

for t in range(200):
    # 1. Measure
    feat, Z = cam.get_image_feature()
    if feat is None: break
    
    # Log (Convert back to pixels for plotting)
    u = feat[0] * cam.fx + cam.cx
    v = feat[1] * cam.fy + cam.cy
    history_u.append(u)
    history_v.append(v)
    
    # 2. Control
    # Goal: Center (0,0 normalized)
    # Assume we estimate Z perfectly 
    v_cmd, err = compute_velocity(feat, Z_est=Z)
    
    # Force Z-velocity to be 0 (Stay in plane for plot simplicity)
    # Or let it converge. L_inv usually mixes Z control.
    # Let's see...
    
    # 3. Actuate
    cam.move(v_cmd[:3]) # Only XYZ velocity
    
    if np.linalg.norm(err) < 0.001:
        print("Converged!")
        break

# Plot Trajectory in Image Plane
plt.plot(history_u, history_v, 'bo-')
plt.plot(320, 240, 'rx', markersize=10, label="Target") # Center
plt.xlim(0, 640)
plt.ylim(480, 0) # Image coords flip Y
plt.grid()
plt.title("Feature Trajectory (IBVS)")
plt.legend()
plt.show()
```

### 3. Expected Output
*   Blue dots trace a line from Initial $(u,v)$ to Center $(320, 240)$.
*   Velocity slows down (exponential decay) as error approaches zero.

---

## 🔬 Lab Exercise: Z-Estimate Error

### 1. Lab Objectives
- Run the simulation with `Z_est = 2.0` (Correct).
- Run again with `Z_est = 1.0` (Underestimate) and `Z_est = 10.0` (Overestimate).
- **Observe:**
    - Underestimate: High Gain. Overshoot/Oscillation.
    - Overestimate: Low Gain. Slow convergence (Sluggish).
- **Stability:** IBVS is robust to Z errors (it will still converge, just weirdly) as long as sign is correct.

---

## 🚀 Project: "USB Insert"

**Goal:** Insert a USB stick into a port.
1.  **Coarse Move:** MoveIt planning to ~5cm away.
2.  **Fine Move (Servoing):**
    - Detect corner features of USB Port.
    - Servo XY to align center.
    - Servo Z to approach ($v_z = 0.5 \text{cm/s}$).
    - Stop when Force Control detects contact.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Field of View Loss"
*   **Symptom:** Optimization moves camera so fast the target leaves the image. Servo crashes (No features).
*   **Fix:** **Gain Scheduling**. Reduce $\lambda$ if $e$ is large. Or add potential field constraints to keep features away from image borders.

#### 2. "Chaotic Rotation"
*   **Symptom:** Camera spirals.
*   **Cause:** Decoupling assumption. Doing large rotation ($90^\circ$) induces large translation in image features (Interaction Matrix coupling). IBVS handles small displacements well, but large ones need PBVS.

---

## ⚡ Optimization: 2.5D Servoing

Hybrid approach.
*   Decompose Homography matrix.
*   Control Rotation ($R_z$) using PBVS features ($\theta$).
*   Control Translation using IBVS features ($u, v$).
*   Guarantees straight line trajectory in space (no weird spiraling).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Does IBVS need robot calibration?
    *   **A:** No (sort of). It needs Eye-to-Hand calibration (Extrinsics), but inherently compensates for kinematic errors because it closes the loop on the *image error*.
2.  **Q:** What if target is too far?
    *   **A:** Pixels are discrete. If feature moves < 1 pixel, error is 0. Deadband limit accuracy. use sub-pixel corner detection.
3.  **Q:** PBVS vs IBVS?
    *   **A:** PBVS moves robot in straight 3D line (Good). IBVS moves features in straight 2D line (Good for keeping target in view).

### Challenge Task
> **Task:** Visual Tracking.
> 1. Target moves (Dynamic).
> 2. Add Feedforward term. $\dot{e} = L_s v_c + \frac{\partial e}{\partial t}$.
> 3. Estimate feature velocity $\frac{\partial e}{\partial t}$ using Kalman Filter and add to control law to reduce lag.

---

## 📚 Further Reading
- **Visual Servo Control:** Chaumette & Hutchinson (IEEE RAM Tutorials).
- **Visp:** Visual Servoing Platform (C++ Library).

---

**Day 55 Complete**
