# Day 75: Visual-Inertial Odometry (VIO)
## Phase 4: ADAS & Robotics Systems | Week 11: Localization

---

> **📝 Day 75 Focus:**
> Wheels slip. GPS gets blocked. But a Camera sees the world, and an IMU feels the motion. **VIO (Visual-Inertial Odometry)** fuses these two to provide robust, drift-free localization for drones, AR headsets, and self-driving cars in tunnels.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** Loosely Coupled (Filter-based) vs Tightly Coupled (Optimization-based) VIO.
2.  **Explain** the role of IMU Preintegration (handling high-rate IMU data).
3.  **Implement** Optical Flow tracking (Lucas-Kanade) for visual constraints.
4.  **Simulate** the VIO pipeline: Feature Tracking + IMU Propagation.
5.  **Identify** key challenges: Initialization, Scale Ambiguity, and Time Synchronization.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 54:** Visual Odometry.
-   **Day 73:** IMU Errors.
-   **Optimization:** Least Squares.

### Hardware Requirements
-   **Camera + IMU:** (Optional) Intel RealSense D435i or Oak-D.

### Software Stack
-   **Python:** `opencv-python`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why VIO?

-   **VO (Visual Only):** Good at slow speed. Fails with fast motion (motion blur) or low texture. Scale ambiguity (Monocular).
-   **IO (Inertial Only):** Good at fast motion. Drifts quickly.
-   **VIO:** Best of both. IMU handles fast jerks and provides scale (gravity). Camera corrects IMU drift.

### 🔹 Part 2: Architecture Types

1.  **MSCKF (Multi-State Constraint Kalman Filter):**
    -   Filter-based (EKF).
    -   State includes sliding window of camera poses.
    -   Fast, efficient. Used in Google ARCore.
2.  **Optimization-Based (VINS-Mono, OKVIS):**
    -   Graph SLAM approach.
    -   Minimizes reprojection error + IMU error.
    -   More accurate, computationally heavy.

### 🔹 Part 3: IMU Preintegration

IMU runs at 200Hz. Camera at 30Hz.
We can't add 200 IMU states to the graph between every keyframe.
**Preintegration:** Combine all IMU measurements between Frame $i$ and $j$ into a single "Relative Motion" constraint $(\Delta p, \Delta v, \Delta q)$.
-   This constraint is independent of the starting state, allowing re-linearization in the optimization.

---

## 💻 Implementation: VIO Frontend (Tracking + Propagation)

**Scenario:**
-   **Input:** Video stream + IMU stream.
-   **Task:** Track features (Visual) and Propagate State (Inertial).
-   **Note:** Full VIO backend (Optimization) is too complex for 100 lines. We focus on the Frontend logic.

### 🛠️ Setup
Create `week11_day75` and `vio_frontend.py`.

```bash
mkdir -p ~/ros2_ws/src/week11_day75
cd ~/ros2_ws/src/week11_day75
touch vio_frontend.py
```

### 👨‍💻 Code: VIO Frontend

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Constants ---
K = np.array([[718.856, 0.0, 607.1928],
              [0.0, 718.856, 185.2157],
              [0.0, 0.0, 1.0]]) # KITTI Calibration
DT_IMU = 0.01 # 100 Hz

class IMUState:
    def __init__(self):
        self.p = np.zeros(3) # Position
        self.v = np.zeros(3) # Velocity
        self.q = np.eye(3)   # Rotation (Matrix)
        self.bg = np.zeros(3) # Gyro Bias
        self.ba = np.zeros(3) # Accel Bias
        self.g = np.array([0, 0, -9.81]) # Gravity

    def propagate(self, accel, gyro, dt):
        # accel, gyro: measurements
        
        # Remove Bias
        a_unbiased = accel - self.ba
        w_unbiased = gyro - self.bg
        
        # Update Rotation (First Order)
        # R_new = R_old * Exp(w * dt)
        # Small angle approximation: Exp(w*dt) ~ I + [w]x * dt
        wx, wy, wz = w_unbiased
        Omega = np.array([[0, -wz, wy],
                          [wz, 0, -wx],
                          [-wy, wx, 0]])
        dR = np.eye(3) + Omega * dt
        self.q = self.q @ dR
        
        # Update Velocity
        # v_new = v_old + (R * a + g) * dt
        acc_world = self.q @ a_unbiased + self.g
        self.v += acc_world * dt
        
        # Update Position
        # p_new = p_old + v * dt + 0.5 * (R * a + g) * dt^2
        self.p += self.v * dt + 0.5 * acc_world * dt**2

class VIOFrontend:
    def __init__(self):
        self.imu_state = IMUState()
        self.prev_img = None
        self.features = None
        self.traj = []
        
    def track_image(self, img):
        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.features is None or len(self.features) < 50:
            # Detect new features
            p = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            if p is not None:
                if self.features is None:
                    self.features = p
                else:
                    self.features = np.vstack((self.features, p))
                    
        # Track existing features (Lucas-Kanade)
        if self.prev_img is not None and len(self.features) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_img, gray, self.features, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            # Select good points
            good_new = p1[st == 1]
            good_old = self.features[st == 1]
            
            # Visualization
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                img = cv2.line(img, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                img = cv2.circle(img, (int(a), int(b)), 5, (0, 0, 255), -1)
                
            self.features = good_new.reshape(-1, 1, 2)
            
        self.prev_img = gray
        return img

    def process_imu(self, accel, gyro):
        self.imu_state.propagate(accel, gyro, DT_IMU)
        self.traj.append(self.imu_state.p.copy())

def main():
    vio = VIOFrontend()
    
    # Simulation: Drone flying forward
    # Camera: 30 FPS
    # IMU: 100 FPS
    
    # Create synthetic image (Starfield)
    width, height = 640, 480
    stars = np.random.randint(0, [width, height], (100, 2))
    
    plt.figure()
    
    for i in range(100): # 100 frames (3.3s)
        # 1. Simulate Motion (Forward X)
        # v = 1 m/s
        # a = 0 (Constant velocity)
        # IMU measures: a = R.T * (a_world - g)
        # If flat: a = [0, 0, 9.81]
        
        # Simulate IMU updates between frames (3 steps)
        for _ in range(3):
            accel = np.array([0.1, 0, 9.81]) # Slight forward accel
            gyro = np.array([0, 0, 0])
            vio.process_imu(accel, gyro)
            
        # 2. Simulate Image
        # Move stars backward (Optical Flow)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        flow_x = -5.0 # Pixels per frame
        stars = stars + [flow_x, 0]
        
        # Reset stars that left screen
        for s in stars:
            if s[0] < 0: s[0] += width
            cv2.circle(img, (int(s[0]), int(s[1])), 3, (255, 255, 255), -1)
            
        # 3. Track
        vis = vio.track_image(img)
        
        # 4. Display
        cv2.imshow('VIO Tracker', vis)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        # Plot Trajectory
        traj = np.array(vio.traj)
        if len(traj) > 0:
            plt.cla()
            plt.plot(traj[:, 0], traj[:, 1], label="IMU Path")
            plt.title("IMU Propagation")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.grid()
            plt.pause(0.01)
            
    cv2.destroyAllWindows()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Drift Check

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   The "IMU Path" shows the drone moving forward (X axis).
    -   The "VIO Tracker" shows green lines (Optical Flow) moving left (stars moving backward).
3.  **Experiment:**
    -   Add bias to the IMU: `accel = np.array([0.1 + 0.1, 0, 9.81])`.
    -   **Result:** The IMU Path will curve or accelerate incorrectly.
    -   **Insight:** In a real VIO, the Visual constraints (stars must move 5px, not 10px) would pull the IMU state back and *estimate* the bias.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Initialization Failure
**Symptom:** VIO flies away immediately.
**Cause:** Gravity vector not aligned. VIO needs to know which way is "Down" to subtract gravity.
**Solution:** Keep stationary for 2 seconds at start (Static Initialization) to estimate gravity and bias.

#### 2. Time Synchronization
**Symptom:** Jittery trajectory.
**Cause:** Camera and IMU clocks are offset.
**Solution:** Use hardware triggering (Camera triggers IMU or vice versa). Or solve for time offset $t_d$ in the optimization.

---

## ⚡ Optimization & Best Practices

### 1. Keyframe Selection
Don't optimize every frame.
-   Select **Keyframes** when the camera moves enough (Parallax) or tracks are lost.
-   Marginalize old keyframes to keep the graph size constant.

### 2. Loop Closure
VIO still drifts over long distances (Loop Drift).
-   Use **DBoW2** (Bag of Words) to detect if you returned to a previous location.
-   Add a Loop Closure constraint to snap the trajectory shut.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the main advantage of VIO over VO?
    *   **A:** Scale. Monocular VO doesn't know if the world is big or small. IMU provides metric scale via gravity and acceleration.
2.  **Q:** What is "Excitation"?
    *   **A:** To estimate IMU bias, the robot must move. If it moves at constant velocity, accel bias is unobservable (looks like drag or gravity error). You need acceleration/rotation to calibrate.
3.  **Q:** Why do we need Preintegration?
    *   **A:** To decouple the IMU integration from the state estimation, allowing us to update the starting state without re-integrating thousands of IMU messages.

### Challenge Task
**Task:** Gravity Removal.
1.  Rotate the IMU (in simulation) by 45 degrees pitch.
2.  Update `accel` to be `R.T @ [0, 0, 9.81]`.
3.  Verify that `propagate` correctly subtracts gravity and shows 0 movement.

---

## 📚 Further Reading & References
-   [VINS-Mono Paper (Qin et al.)](https://arxiv.org/abs/1708.03852)
-   [OpenVINS (State of the art VIO)](https://github.com/rpng/open_vins)

---

**Day 75 Complete** | Phase 4: ADAS & Robotics Systems | Week 11: Localization
