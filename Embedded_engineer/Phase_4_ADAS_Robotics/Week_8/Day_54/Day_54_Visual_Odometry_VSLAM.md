# Day 54: Visual Odometry (VO) & VSLAM
## Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion

---

> **📝 Day 54 Focus:**
> Wheel encoders slip on ice. GPS fails in tunnels. But eyes (Cameras) work everywhere. **Visual Odometry (VO)** is the art of estimating how far you've moved by analyzing how the world moves in your camera feed. Today, we build a VO pipeline from scratch using OpenCV.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Monocular VO pipeline: Features -> Matching -> Essential Matrix -> Pose.
2.  **Extract** ORB features and match them between consecutive frames.
3.  **Compute** the Essential Matrix ($E$) and recover Rotation ($R$) and Translation ($t$).
4.  **Implement** a VO system in Python to track a vehicle's trajectory.
5.  **Understand** the scale ambiguity problem in Monocular VO.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Computer Vision:** Pinhole Camera Model, Intrinsic Matrix ($K$).
-   **Linear Algebra:** SVD (Singular Value Decomposition).

### Hardware Requirements
-   **None:** We will use a synthetic dataset or standard video.

### Software Stack
-   **Python:** `opencv-python`, `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Geometry of Motion

When a camera moves, the pixels move.
-   **Epipolar Geometry:** Relates two views of the same scene.
-   **Essential Matrix ($E$):** Encodes the relative pose ($R, t$) between two calibrated views.
    -   $x_2^T E x_1 = 0$ (Epipolar Constraint).
    -   $x_1, x_2$: Normalized image coordinates.

### 🔹 Part 2: The Pipeline

1.  **Capture:** Frame $I_{t-1}$ and $I_t$.
2.  **Detect:** Find "corners" (Features) in both frames (ORB/FAST).
3.  **Match:** Find which corner in $I_t$ corresponds to $I_{t-1}$ (BFMatcher/Optical Flow).
4.  **Recover Pose:**
    -   Calculate $E$ using RANSAC (to ignore outliers).
    -   Decompose $E$ into $R$ and $t$.
5.  **Update:** $P_t = P_{t-1} \cdot T_t$.

### 🔹 Part 3: Scale Ambiguity

With one camera (Monocular), you can tell the *direction* of motion, but not the *magnitude*.
-   Did I move 1 meter forward in a small room?
-   Or 10 meters forward in a giant room?
-   **Solution:** Use Stereo Cameras, IMU fusion, or known object sizes.

---

## 💻 Implementation: Monocular VO

**Scenario:**
-   We simulate a camera moving forward.
-   We generate 3D points (World).
-   We project them into 2D frames.
-   We run VO to recover the trajectory.

### 🛠️ Setup
Create `week8_day54` and `visual_odometry.py`.

```bash
mkdir -p ~/ros2_ws/src/week8_day54
cd ~/ros2_ws/src/week8_day54
touch visual_odometry.py
```

### 👨‍💻 Code: VO Pipeline

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

class VisualOdometry:
    def __init__(self, K):
        self.K = K # Intrinsic Matrix
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        self.prev_kp = None
        self.prev_des = None
        
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        
        self.traj_x = []
        self.traj_z = []

    def process_frame(self, img):
        # 1. Detect & Compute
        kp, des = self.orb.detectAndCompute(img, None)
        
        if self.prev_kp is None:
            self.prev_kp = kp
            self.prev_des = des
            return
        
        # 2. Match
        matches = self.bf.knnMatch(self.prev_des, des, k=2)
        
        # Lowe's Ratio Test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
                
        if len(good) < 5:
            return # Lost tracking
            
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good])
        
        # 3. Find Essential Matrix
        E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # 4. Recover Pose
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K)
        
        # Update Trajectory
        # Note: t is a unit vector (Scale Ambiguity). We assume scale=1.0 for simplicity.
        # In real MonoVO, we need an external scale source.
        scale = 1.0 
        
        if (t[2] > t[0]) and (t[2] > t[1]): # Forward motion check
            self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
            self.cur_R = self.cur_R.dot(R)
            
        self.traj_x.append(self.cur_t[0][0])
        self.traj_z.append(self.cur_t[2][0])
        
        # Update Previous
        self.prev_kp = kp
        self.prev_des = des
        
        # Draw Matches (Optional)
        img_matches = cv2.drawMatches(img, self.prev_kp, img, kp, good[:20], None, flags=2)
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(1)

def run_simulation():
    # Camera Intrinsics (Simulated)
    W, H = 640, 480
    f = 500
    K = np.array([[f, 0, W/2],
                  [0, f, H/2],
                  [0, 0, 1]])
                  
    vo = VisualOdometry(K)
    
    # Generate Synthetic World (Random 3D points)
    points_3d = np.random.rand(1000, 3) * 100 - 50 # X, Y in [-50, 50]
    points_3d[:, 2] += 10 # Z in [10, 110]
    
    # Simulate Camera Motion (Moving forward in Z)
    traj_true_x = []
    traj_true_z = []
    
    for i in range(100):
        # Move Camera: Z decreases by 1.0
        # Equivalent to Points moving Z decreases by 1.0 relative to camera
        # Actually, let's move camera: t_cam = [0, 0, i]
        # Points relative to camera: P_cam = P_world - t_cam
        
        t_cam = np.array([0, 0, i * 1.0])
        traj_true_x.append(t_cam[0])
        traj_true_z.append(t_cam[2])
        
        # Project Points
        pts_cam = points_3d - t_cam
        
        # Filter points behind camera
        valid = pts_cam[:, 2] > 1.0
        pts_cam = pts_cam[valid]
        
        # Project to 2D
        u = (pts_cam[:, 0] * f / pts_cam[:, 2]) + W/2
        v = (pts_cam[:, 1] * f / pts_cam[:, 2]) + H/2
        
        # Create Image (Black background, White dots)
        img = np.zeros((H, W), dtype=np.uint8)
        for j in range(len(u)):
            if 0 <= u[j] < W and 0 <= v[j] < H:
                cv2.circle(img, (int(u[j]), int(v[j])), 3, 255, -1)
                
        # Run VO
        vo.process_frame(img)
        
    # Plot
    plt.figure()
    plt.plot(traj_true_x, traj_true_z, label='True')
    plt.plot(vo.traj_x, vo.traj_z, label='Est (VO)')
    plt.legend()
    plt.title("Visual Odometry Trajectory")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.show()

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: The Scale Problem

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** The estimated trajectory shape is correct (Straight line), but the *length* might be perfect (because we hardcoded `scale=1.0`) or wrong if we change the speed.
3.  **Experiment:**
    -   Change the camera speed to `i * 2.0` (2m/s).
    -   Keep `scale = 1.0` in the VO code.
    -   *Result:* The VO will still say we moved 100 units (100 frames * 1.0), but truth is 200 units.
    -   **Conclusion:** Monocular VO cannot measure speed/distance without external info.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. RANSAC Fails
**Symptom:** Trajectory jumps wildly.
**Cause:** Too few features or too many outliers (moving objects).
**Solution:** Increase feature count (3000). Use Optical Flow (KLT) instead of matching for smoother tracking.

#### 2. Pure Rotation
**Symptom:** $E$ matrix calculation is unstable.
**Cause:** If the camera only rotates (no translation), $E$ is undefined (Baseline is 0).
**Solution:** Use Homography ($H$) instead of Essential Matrix ($E$) for pure rotation or planar scenes.

---

## ⚡ Optimization & Best Practices

### 1. Keyframes
Don't process every frame.
-   If $I_t$ is too similar to $I_{t-1}$, the baseline is too small -> Error.
-   Wait until the camera has moved enough, then create a **Keyframe**.
-   Match current frame against the last Keyframe.

### 2. Bundle Adjustment (BA)
VO accumulates drift (Dead Reckoning).
**Local BA:** Optimize the last $N$ poses and 3D points to minimize reprojection error.
-   Used in systems like **ORB-SLAM**.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we need the Intrinsic Matrix $K$?
    *   **A:** To convert pixel coordinates (u, v) into normalized coordinates (x, y) that represent physical rays. $E$ works on rays, not pixels.
2.  **Q:** What is the difference between VO and VSLAM?
    *   **A:** VO focuses on the trajectory (Local consistency). VSLAM focuses on the map and loop closure (Global consistency).
3.  **Q:** How does Stereo VO solve scale ambiguity?
    *   **A:** The baseline $b$ between the two cameras is known and fixed. This provides a metric scale reference.

### Challenge Task
**Task:** Optical Flow VO.
1.  Replace ORB matching with `cv2.calcOpticalFlowPyrLK`.
2.  Track points from frame to frame.
3.  This is faster and often more robust for video sequences.

---

## 📚 Further Reading & References
-   [Scaramuzza's VO Tutorial](http://rpg.ifi.uzh.ch/visual_odometry_tutorial.html)
-   [ORB-SLAM3 Paper](https://arxiv.org/abs/2008.00854)

---

**Day 54 Complete** | Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion
