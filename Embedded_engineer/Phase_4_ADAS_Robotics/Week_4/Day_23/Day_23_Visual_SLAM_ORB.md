# Day 23: Visual SLAM (ORB-SLAM)
## Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)

---

> **📝 Day 23 Focus:**
> Lidar is precise but expensive. Cameras are cheap and ubiquitous. **Visual SLAM** allows us to build 3D maps and track motion using only video. Today, we dissect **ORB-SLAM**, the gold standard for sparse visual SLAM, and implement a basic Visual Odometry pipeline.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** Monocular, Stereo, and RGB-D SLAM (Scale Ambiguity).
2.  **Explain** the three threads of ORB-SLAM: Tracking, Local Mapping, and Loop Closing.
3.  **Understand** Keyframes, Map Points, and the Covisibility Graph.
4.  **Implement** a Monocular Visual Odometry (VO) system using OpenCV (Feature Matching + 5-Point Algorithm).
5.  **Visualize** the estimated trajectory against ground truth.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 15:** Camera Calibration ($K$).
-   **Day 16:** ORB Features.
-   **Day 17:** Optical Flow (KLT).

### Hardware Requirements
-   **Dataset:** KITTI Odometry Dataset (Sequence 00) is standard. We will use a sample video.

### Software Stack
-   **Python Libraries:** `opencv-python`, `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Visual SLAM Paradigms

#### 1.1 Monocular SLAM
-   **Input:** Single camera.
-   **Challenge:** **Scale Ambiguity**. You can't tell the difference between a small room and a giant castle. The map is "up to scale".
-   **Solution:** Need IMU fusion or known object size to recover scale.

#### 1.2 Stereo SLAM
-   **Input:** Two calibrated cameras.
-   **Benefit:** Depth is directly observable via disparity ($Z = \frac{f \cdot B}{d}$). No scale ambiguity.

#### 1.3 RGB-D SLAM
-   **Input:** Color + Depth (Kinect/RealSense).
-   **Benefit:** Dense maps, easy initialization. Limited range (indoor).

---

### 🔹 Part 2: ORB-SLAM Architecture

ORB-SLAM (Mur-Artal et al.) is famous for its robust 3-thread architecture.

#### 2.1 Tracking Thread (Real-Time)
-   Extracts ORB features from current frame.
-   Matches with previous frame (or local map).
-   Estimates Pose ($R, t$) using **PnP (Perspective-n-Point)**.
-   Decides if a new **Keyframe** is needed.

#### 2.2 Local Mapping Thread
-   Takes new Keyframes.
-   Triangulates new **Map Points** (3D landmarks).
-   Performs **Local Bundle Adjustment** (Optimizes recent Keyframes and Points).
-   Culls bad points.

#### 2.3 Loop Closing Thread
-   Checks if current Keyframe looks like an old one (using Bag of Words).
-   If loop detected:
    -   Computes similarity transform (Sim3).
    -   Fuses duplicate points.
    -   Performs **Pose Graph Optimization** to correct drift.

---

### 🔹 Part 3: Bundle Adjustment (BA)

BA is the core optimization engine.
It minimizes the **Reprojection Error**: The distance between the observed 2D feature and the projected 3D map point.

$$ \min_{R_i, t_i, X_j} \sum_{i,j} || u_{ij} - \pi(R_i X_j + t_i) ||^2 $$

-   $R_i, t_i$: Pose of camera $i$.
-   $X_j$: Position of 3D point $j$.
-   $\pi$: Projection function (Pinhole).

---

## 💻 Implementation: Monocular Visual Odometry

We will build a simplified VO pipeline:
1.  Detect features in Frame $t-1$.
2.  Track them to Frame $t$ using Optical Flow.
3.  Compute Essential Matrix ($E$) using Nister's 5-point algorithm.
4.  Recover $R, t$ from $E$.
5.  Accumulate trajectory.

### 🛠️ Setup
Create `week4_day23` and `visual_odometry.py`.

```bash
mkdir -p ~/ros2_ws/src/week4_day23
cd ~/ros2_ws/src/week4_day23
touch visual_odometry.py
```

### 👨‍💻 Code: Visual Odometry Class

```python
import numpy as np
import cv2
import os

class VisualOdometry:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = sorted(os.listdir(data_dir))
        
        # Camera Intrinsics (KITTI Sequence 00)
        self.focal = 718.8560
        self.pp = (607.1928, 185.2157)
        self.K = np.array([[self.focal, 0, self.pp[0]],
                           [0, self.focal, self.pp[1]],
                           [0, 0, 1]])
                           
        # Feature Detector params
        self.feature_params = dict(maxCorners=2000, qualityLevel=0.01, minDistance=10, blockSize=3)
        self.lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # State
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.traj = []

    def process_frame(self, i):
        if i == 0:
            self.prev_img = cv2.imread(os.path.join(self.data_dir, self.images[0]), 0)
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_img, mask=None, **self.feature_params)
            return

        curr_img = cv2.imread(os.path.join(self.data_dir, self.images[i]), 0)
        
        # Track features
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_img, curr_img, self.prev_pts, None, **self.lk_params)
        
        # Filter good points
        good_prev = self.prev_pts[status == 1]
        good_curr = curr_pts[status == 1]
        
        if len(good_prev) < 50: # Lost track? Re-detect
             curr_pts = cv2.goodFeaturesToTrack(curr_img, mask=None, **self.feature_params)
             self.prev_img = curr_img
             self.prev_pts = curr_pts
             return

        # Compute Essential Matrix
        E, mask = cv2.findEssentialMat(good_curr, good_prev, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0)
        
        # Recover Pose
        _, R, t, mask = cv2.recoverPose(E, good_curr, good_prev, focal=self.focal, pp=self.pp)
        
        # Scale Check (Monocular Scale Ambiguity)
        # In real MonoSLAM, we don't know scale. 
        # Here, we can cheat using Ground Truth speed, or just assume scale=1 (drift will be huge).
        # For this demo, we assume scale = 1.0
        scale = 1.0 
        
        if scale > 0.1: # Only update if moving
            self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
            self.cur_R = self.cur_R.dot(R)
            
        self.traj.append((self.cur_t[0][0], self.cur_t[2][0])) # X, Z (Forward)
        
        # Update
        self.prev_img = curr_img
        self.prev_pts = good_curr.reshape(-1, 1, 2)
        
        # Re-detect if too few features
        if len(self.prev_pts) < 1000:
            new_pts = cv2.goodFeaturesToTrack(self.prev_img, mask=None, **self.feature_params)
            self.prev_pts = np.concatenate((self.prev_pts, new_pts), axis=0)

    def draw_trajectory(self, img_size=(800, 800)):
        traj_img = np.zeros(img_size, dtype=np.uint8)
        
        for i, (x, z) in enumerate(self.traj):
            # Map coordinates to image
            draw_x = int(x) + 400
            draw_y = int(z) + 100
            cv2.circle(traj_img, (draw_x, draw_y), 1, (255, 255, 255), 1)
            
        cv2.imshow('Trajectory', traj_img)
        cv2.waitKey(1)

def run_vo():
    # You need to download KITTI sequence 00 grayscale images
    # Path: dataset/sequences/00/image_0/
    data_path = 'path/to/kitti/00/image_0' 
    
    if not os.path.exists(data_path):
        print("Please set the correct path to KITTI dataset")
        return

    vo = VisualOdometry(data_path)
    
    for i in range(len(vo.images)):
        vo.process_frame(i)
        vo.draw_trajectory()

if __name__ == "__main__":
    run_vo()
```

---

## 🔬 Lab Exercise: Scale Drift

### Lab Objectives
1.  Run the VO script on a video.
2.  **Observation:** The shape of the path looks correct (turns are correct), but the *length* of the segments is arbitrary.
3.  **Experiment:** If you have ground truth speed $v_{gt}$ (e.g., from CAN bus), set `scale = v_gt * dt`.
    -   *Result:* The trajectory will match the real world much better. This is how production Mono-VO works (Camera + CAN).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Pure Rotation
**Symptom:** Essential Matrix calculation fails or returns garbage.
**Cause:** $E$ is undefined for pure rotation (no translation).
**Solution:** Use Homography ($H$) instead of $E$ when translation is near zero. ORB-SLAM automatically switches between $E$ and $H$ models.

#### 2. Feature Starvation
**Symptom:** Tracking lost in sky or textureless road.
**Cause:** Not enough corners.
**Solution:** Adaptive thresholding for feature detection. Ensure features are distributed across the image (Grid-based detection).

#### 3. Scale Drift
**Symptom:** The map shrinks or expands over time.
**Cause:** Errors in $t$ estimation accumulate multiplicatively.
**Solution:** Loop Closure is the only way to fix this in pure SLAM.

---

## ⚡ Optimization & Best Practices

### 1. Keyframes
Don't triangulate points every frame. Only add a Keyframe when:
-   Enough time has passed.
-   The camera has moved significantly.
-   Tracking quality drops (need new features).

### 2. Covisibility Graph
Maintain a graph where nodes are Keyframes and edges represent shared Map Points.
-   Allows efficient Local Bundle Adjustment (only optimize connected Keyframes).

### 3. Bag of Words (DBoW2)
Convert image descriptors into a "Word Vector".
-   Allows $O(1)$ lookup for Loop Closure ("Have I seen this place before?").

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Essential Matrix"?
    *   **A:** A 3x3 matrix that encodes the relative pose ($R, t$) between two calibrated views. $x'^T E x = 0$.
2.  **Q:** Why is Monocular SLAM "Scale Ambiguous"?
    *   **A:** A large motion in a large world looks identical to a small motion in a small world on the image plane.
3.  **Q:** What is the difference between VO and SLAM?
    *   **A:** VO integrates path (drifts). SLAM builds a map and closes loops (corrects drift).

### Challenge Task
**Task:** Stereo VO.
1.  Use `cv2.stereoRectify` and `cv2.StereoBM` to compute a disparity map.
2.  For each feature, get its depth $Z$.
3.  Use `cv2.solvePnPRansac` (3D-to-2D) instead of `findEssentialMat` (2D-to-2D).
    -   *Result:* Absolute scale! No drift in size.

---

## 📚 Further Reading & References
-   [ORB-SLAM Paper](https://arxiv.org/abs/1502.00956)
-   [Visual Odometry Tutorial (Scaramuzza)](http://rpg.ifi.uzh.ch/visual_odometry_tutorial.html)

---

**Day 23 Complete** | Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)
