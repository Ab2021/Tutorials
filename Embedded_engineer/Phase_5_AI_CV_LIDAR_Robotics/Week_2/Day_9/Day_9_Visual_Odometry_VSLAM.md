# Day 9: Visual Odometry & VSLAM
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 2: Advanced SLAM & State Estimation

---

> **📝 Content Creator Instructions:**
> Cameras are cheap, passive, and rich in information. But 3D from 2D is hard.
> - **Focus:** Feature tracking, Epipolar Geometry, and the ORB-SLAM architecture.
> - **Code:** Implementation of Monocular Visual Odometry from scratch.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Sparse (Feature-based) and Dense (Direct) VO.
2.  **Compute** the Essential Matrix ($E$) and recover Camera Pose ($R, t$) from point matches.
3.  **Implement** a Tracking frontend using Lucas-Kanade Optical Flow.
4.  **Explain** Local Bundle Adjustment and its role in reducing drift.
5.  **Run** ORB-SLAM3 on a monocular video sequence.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Webcam or Video Dataset (KITTI Sample 00.mp4).

### Software Environment
```bash
pip install opencv-python numpy matplotlib
# For ORB-SLAM3 (Advanced Lab):
# Requires C++, Pangolin, Eigen, g2o
```

### Prior Knowledge
- Pinhole Camera Model ($K$ matrix).
- Projective Geometry.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Visual Odometry Fundamentals

**Goal:** Estimate trajectory $T_k, T_{k+1}, ...$ from a stream of images $I_k, I_{k+1}$.

#### 1.1 The Epipolar Constraint
For a 3D point $P$ observed as $p_1$ in Image 1 and $p_2$ in Image 2:
$$ p_2^T E p_1 = 0 $$
Where $E$ is the **Essential Matrix**: $E = [t]_{\times} R$.
*   We can compute $E$ from 5 corresponding points (Nister's Algorithm) or 8 points (Longuet-Higgins Algorithm).
*   Once we have $E$, we decompose it (SVD) to get Rotation $R$ and Translation $t$ (up to scale).

#### 1.2 Scale Ambiguity (Monocular)
With one camera, we cannot know the valid scale. A tiny house close up looks the same as a huge house far away.
*   **Result:** Trajectory is correct shape, but arbitrary size.
*   **Fix:** Use Stereo Camera, IMU constrained (VINS), or known object size.

### 🔹 Part 2: ORB-SLAM Architecture

The current gold standard for sparse VSLAM.

1.  **Tracking (Frontend):**
    *   Extract ORB features (Fast corners + BRIEF descriptors).
    *   Match with previous frame.
    *   Minimize Reprojection Error to refine pose.

2.  **Local Mapping (Backend):**
    *   Triangulate new 3D points (MapPoints) from matched keyframes.
    *   **Local Bundle Adjustment (BA):** Jointly optimize the last $N$ Keyframes and all MapPoints seen by them.
    *   Minimize $\sum || p_{obs} - \pi(C_i, X_j) ||^2$.

3.  **Loop Closing:**
    *   Use Bag of Words (BoW) to recognize a place previously visited.
    *   **Sim3 Optimization:** Correct scale drift if loop is detected.

---

## 💻 Implementation: Mono-VO from Scratch

We will implement a simple VO pipeline:
1.  Detect params.
2.  Track via KLT Optical Flow.
3.  Compute Essential Matrix.
4.  Recover Pose.

### 🛠️ Project Structure
```text
day9_vo/
├── data/
│   └── 00.mp4 (KITTI)
├── src/
│   ├── visual_odometry.py
│   └── calibration.py
└── run_pipeline.py
```

### 👨‍💻 Code Implementation (`src/visual_odometry.py`)

```python
import numpy as np
import cv2

class VisualOdometry:
    def __init__(self, cam_intrinsics):
        self.K = cam_intrinsics
        self.pp = (self.K[0, 2], self.K[1, 2]) # Principal Point
        self.focal = self.K[0, 0]
        
        self.lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        
        self.prev_img = None
        self.prev_pts = None
        
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        
    def process_frame(self, img_bgr):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.prev_img is None:
            # First Frame initialization
            self.prev_img = img_gray
            self.prev_pts = self.detector.detect(img_gray, None)
            self.prev_pts = np.array([x.pt for x in self.prev_pts], dtype=np.float32).reshape(-1, 1, 2)
            return self.cur_t
            
        # 1. Track Points (Optical Flow)
        cur_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img_gray, self.prev_pts, None, **self.lk_params)
        
        # Filter valid points
        good_old = self.prev_pts[status == 1]
        good_new = cur_pts[status == 1]
        
        if len(good_new) < 100:
             # Redetect features if lost
             self.prev_pts = self.detector.detect(img_gray, None)
             self.prev_pts = np.array([x.pt for x in self.prev_pts], dtype=np.float32).reshape(-1, 1, 2)
             self.prev_img = img_gray
             return self.cur_t

        # 2. Compute Essential Matrix
        E, mask = cv2.findEssentialMat(good_new, good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0)
        
        # 3. Recover Pose
        _, R, t, mask = cv2.recoverPose(E, good_new, good_old, self.K)
        
        # Update Trajectory (t is a unit vector, need absolute scale from somewhere)
        # Hack: Since KITTI is a car, assume scale is related to speed or const 1.0
        absolute_scale = 1.0 # In real mono-vo, this is the main problem
        
        if absolute_scale > 0.1:
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = self.cur_R.dot(R)
            
        # Update State
        self.prev_img = img_gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        
        return self.cur_t
```

---

## 🔬 Lab Exercise: ORB-SLAM3 in Docker

### 1. Lab Objectives
- Run the state-of-the-art ORB-SLAM3 system on a pre-recorded dataset.
- Observe "Loop Closure" in action.

### 2. Step-by-Step Guide

#### Phase A: Build & Run

```bash
# Docker Command
docker run -it --rm \
    -v /path/to/kitti/dataset:/dataset \
    --network host \
    orbslam3_ros2:humble
    
# Launch Node (Monocular)
ros2 run orbslam3 mono /orb_slam3/Vocabulary/ORBvoc.txt /orb_slam3/Examples/Monocular/KITTI00-02.yaml
```

#### Phase B: Visualization
Rviz will show the sparse map (white dots) and the camera frustum.
*   **Drift:** Watch the robot drive in a loop. When it returns to start, does the map snap together? That is Loop Closure.

---

## 🚀 Project: "Home Surveillance VO"

**Goal:** Walk around your room with a Webcam. Map the trajectory.
**Challenge:** Pure rotation (looking around without moving feet) kills Monocular VO (Need translation to triangulate).

### 1. Robustness Logic
Modify the `VisualOdometry` class to detect "Pure Rotation".
*   If `t` is small but `R` is large -> Do NOT update translation, only rotation.
*   Or use a "Homography" model instead of "Essential Matrix" when scene is planar or motion is purely rotational.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Trajectory is a Straight Line"
*   **Cause:** Epipolar geometry (Essential Matrix) fails for pure forward motion (Center of expansion).
*   **Fix:** Use multi-frame tracking or switch to 3D-2D PnP (Perspective-n-Point) if you have an initial map.

#### 2. "Scale Drift"
*   **Symptom:** The world seems to shrink or grow.
*   **Cause:** Monocular scale ambiguity errors accumulate.
*   **Fix:** Detect ground plane (assume height = 1.6m) to constantly correct scale.

---

## ⚡ Optimization: SimD & Feature Pyramids

*   **FAST Corners:** Are fast due to SIMD instructions (checking pixel circle brightness).
*   **Pyramids:** Processing tracking at coarse level (small image) first handles large motions (fast camera movement).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the minimum number of point correspondences needed to solve for $E$?
    *   **A:** 5 (Nister's 5-point algorithm), but 8 is linear and simpler (8-point algorithm).
2.  **Q:** Why does Monocular VO fail on pure rotation?
    *   **A:** Without translation (baseline), there is no parallax. Depth cannot be triangulated. The Essential Matrix becomes undefined (rank deficient).
3.  **Q:** What does "Bundle Adjustment" adjust?
    *   **A:** It adjusts the "Bundle" of rays. It optimizes Camera Poses ($C_i$) and 3D Points ($X_j$) simultaneously to minimize reprojection error.

### Challenge Task
> **Task:** Implement Stereo VO.
> 1. Use `cv2.StereoBM` to compute Disparity.
> 2. Calculate Depth $Z = \frac{f \cdot B}{d}$.
> 3. Now you match 3D points to 3D points (Point-to-Point ICP) or 3D-to-2D (PnP).
> 4. Verify that Scale is now correct metric (meters).

---

## 📚 Further Reading
- **ORB-SLAM:** Mur-Artal et al. (IEEE TRO 2015).
- **SVO (Semi-Direct VO):** Forster et al. (ICRA 2014).
- **Visual-Inertial:** VINS-Mono (Qin et al., 2018).

---

**Day 9 Complete**
