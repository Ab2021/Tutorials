# Day 129: Visual Odometry (Feature-based)
## Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM

---

> **📝 Day 129 Focus:**
> GPS is blocked in tunnels. IMU drifts. **Visual Odometry (VO)** uses the camera to track motion. By watching how feature points move in the image, we can calculate how the car moved in 3D.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the VO pipeline: Features -> Matching -> Motion Estimation.
2.  **Extract** ORB features and match them between frames.
3.  **Calculate** the Essential Matrix ($E$) and recover Pose ($R, t$).
4.  **Implement** a Monocular VO pipeline in OpenCV.
5.  **Visualize** the estimated trajectory.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Computer Vision:** Pinhole Camera Model.
-   **Linear Algebra:** SVD (Singular Value Decomposition).

### Hardware Requirements
-   **None:** Video dataset required (KITTI sequence 00 is standard).

### Software Stack
-   **Python:** `opencv-python`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Pipeline

1.  **Feature Extraction:** Find "corners" (ORB/FAST) in Frame $t-1$ and Frame $t$.
2.  **Feature Matching:** Find corresponding points.
3.  **Motion Estimation:**
    -   **2D-2D:** Epipolar Geometry (Essential Matrix). Used for Monocular initialization.
    -   **3D-2D:** PnP (Perspective-n-Point). Used when we have a map.
    -   **3D-3D:** ICP (Iterative Closest Point). Used for Stereo/RGB-D.

### 🔹 Part 2: Epipolar Geometry

Given two images of the same scene:
$$ x_2^T E x_1 = 0 $$
-   $x_1, x_2$: Normalized coordinates of matched points.
-   $E$: Essential Matrix ($3 \times 3$). Encodes Rotation ($R$) and Translation ($t$).
-   We solve for $E$ using the **5-Point Algorithm** (Nister) or 8-Point Algorithm.
-   Decompose $E \to R, t$.

### 🔹 Part 3: Scale Ambiguity

In Monocular VO, we cannot know the *scale*.
-   Did we move 1 meter forward, or is the world just 2x smaller?
-   We need an external reference (Speedometer, IMU, or known object height) to fix the scale.

---

## 💻 Implementation: MonoVO

**Scenario:**
-   Input: Video of driving.
-   Task: Plot the path.

### 🛠️ Setup
Create `week19_day129` and `visual_odometry.py`.
Download a sample video or use a synthetic one.

```bash
mkdir -p ~/ros2_ws/src/week19_day129
cd ~/ros2_ws/src/week19_day129
touch visual_odometry.py
```

### 👨‍💻 Code: ORB-based VO

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

class VisualOdometry:
    def __init__(self, focal_length, pp):
        self.focal = focal_length
        self.pp = pp # Principal Point (cx, cy)
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        self.prev_img = None
        self.prev_kp = None
        self.prev_des = None
        
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = []

    def process_frame(self, img):
        # 1. Detect and Compute
        kp, des = self.orb.detectAndCompute(img, None)
        
        if self.prev_img is None:
            self.prev_img = img
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
                
        if len(good) < 50:
            print("Not enough matches")
            return
            
        # Extract points
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good])
        
        # 3. Find Essential Matrix
        E, mask = cv2.findEssentialMat(pts2, pts1, focal=self.focal, pp=self.pp, 
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # 4. Recover Pose
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, focal=self.focal, pp=self.pp)
        
        # Scale Ambiguity: Assume constant speed 1.0 (or read from CAN bus)
        scale = 1.0 
        
        # Update Trajectory
        # t is the translation from Frame 2 to Frame 1 in Frame 2's coords.
        # Absolute Pose Update: T_world_new = T_world_old * T_old_new
        
        if scale > 0.1: # Only update if moving
            self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
            self.cur_R = self.cur_R.dot(R)
            
        self.trajectory.append((self.cur_t[0,0], self.cur_t[2,0])) # X, Z (Forward)
        
        # Update previous
        self.prev_img = img
        self.prev_kp = kp
        self.prev_des = des
        
        # Draw Matches
        img_matches = cv2.drawMatches(self.prev_img, self.prev_kp, img, kp, good[:20], None)
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(1)

def main():
    # Camera Intrinsics (Approx for 640x480)
    focal = 718.8560
    pp = (607.1928, 185.2157)
    
    vo = VisualOdometry(focal, pp)
    
    # Load Video (Replace with path to KITTI sequence)
    # cap = cv2.VideoCapture('kitti_00.mp4')
    
    # For demo, we generate synthetic flow
    print("Starting VO (Synthetic Demo)...")
    
    traj_x = []
    traj_z = []
    
    # Simulate a camera moving forward
    # In a real script, read frames from 'cap'
    # Here we just explain the loop structure.
    print("Please run this with a real video file for results.")
    print("Example: Download KITTI Odometry Sequence 00")
    
    # Placeholder for plotting logic
    plt.figure()
    plt.title("Estimated Trajectory (XZ)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Scale Drift

### Lab Objectives
1.  Run the script with a real video (e.g., `test_countryroad.mp4`).
2.  **Observation:** The shape of the path looks correct (turns left when car turns left).
3.  **Issue:** The scale is wrong. The car might travel "100 units", but is it meters? inches?
4.  **Experiment:**
    -   Hardcode `scale = 0.0`.
    -   **Result:** The trajectory stays at (0,0). Rotation still updates.
    -   **Lesson:** Monocular VO is great for orientation, but needs help for translation.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Pure Rotation
**Symptom:** $E$ matrix calculation fails or gives garbage.
**Cause:** If the camera only rotates (no translation), Epipolar Geometry is degenerate.
**Solution:** Use Homography ($H$) instead of Essential Matrix ($E$) for pure rotation or planar scenes. Modern VO switches between $E$ and $H$ automatically.

#### 2. Feature Loss
**Symptom:** "Not enough matches".
**Cause:** Low texture (white wall, sky) or motion blur.
**Solution:** Use more robust features (SIFT/SURF - slower) or Optical Flow (KLT Tracker).

---

## ⚡ Optimization & Best Practices

### 1. Bundle Adjustment (BA)
VO accumulates error (Drift).
-   **Local BA:** Optimize the last $N$ poses and points to minimize reprojection error.
-   Refines the trajectory and reduces drift.

### 2. Keyframes
Don't process every frame.
-   If Frame $t$ and Frame $t+1$ are too similar, skip $t+1$.
-   Only add a **Keyframe** when the camera has moved enough.
-   Saves computation and improves triangulation accuracy (wider baseline).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the Essential Matrix?
    *   **A:** A $3 \times 3$ matrix that relates normalized points in two views. It encapsulates the rotation and translation (up to scale).
2.  **Q:** Why do we use RANSAC?
    *   **A:** To reject outliers. Some feature matches will be wrong (e.g., a cloud moving in the sky). RANSAC finds the model that fits the majority of inliers.
3.  **Q:** How does Stereo VO differ?
    *   **A:** Stereo cameras have a known baseline. We can calculate absolute depth and absolute scale directly. No scale ambiguity.

### Challenge Task
**Task:** Optical Flow VO.
1.  Replace ORB matching with `cv2.calcOpticalFlowPyrLK`.
2.  Track points from frame to frame.
3.  This is faster and often smoother than descriptor matching.

---

## 📚 Further Reading & References
-   [Scaramuzza's VO Tutorial](https://rpg.ifi.uzh.ch/visual_odometry_tutorial.html)
-   [ORB-SLAM Paper](https://arxiv.org/abs/1502.00956)

---

**Day 129 Complete** | Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM
