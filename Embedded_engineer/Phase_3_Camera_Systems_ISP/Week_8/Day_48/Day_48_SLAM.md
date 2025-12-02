# Day 48: SLAM (Simultaneous Localization and Mapping)
## Phase 3: Camera Systems & ISP | Week 8: Depth Sensing & 3D Vision

---

## 🎯 Learning Objectives
1.  **Understand** the SLAM problem: "Where am I?" (Localization) + "What does the world look like?" (Mapping).
2.  **Implement** Visual Odometry (VO) to estimate trajectory from frame-to-frame motion.
3.  **Analyze** the components of a SLAM system: Tracking, Local Mapping, Loop Closure.
4.  **Study** ORB-SLAM2/3 architecture (Feature-based).
5.  **Debug** tracking loss and scale drift issues.
6.  **Visualize** camera trajectory and sparse map points.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Monocular or Stereo Camera.
*   **Software:** OpenCV, g2o (Graph Optimization), ORB-SLAM2 (optional reference).
*   **Knowledge:** Rigid Body Transformations (SE3), Bundle Adjustment.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The SLAM Problem
*   **Localization:** Estimating the camera Pose ($R, t$) at time $t$.
*   **Mapping:** Estimating the 3D positions of landmarks ($X_i$) in the world.
*   **Chicken and Egg:** To localize, you need a map. To map, you need to know where you are. SLAM solves both simultaneously.

### 🔹 Part 2: Visual Odometry (VO)
*   Estimates motion between consecutive frames ($t$ and $t+1$).
*   **Method:**
    1.  Detect Features in Frame $t-1$.
    2.  Track Features to Frame $t$ (Optical Flow or Matching).
    3.  Solve Perspective-n-Point (PnP) or Essential Matrix ($E$) decomposition to find $R, t$.
*   **Drift:** Errors accumulate over time. If you walk in a circle, the estimated end point won't match the start point.

### 🔹 Part 3: Loop Closure
*   **Concept:** Recognizing a place you've visited before.
*   **Action:** When a loop is detected, the system calculates the accumulated error (Drift) and distributes it across the entire path (Pose Graph Optimization). This "snaps" the trajectory shut.

---

## 💻 Implementation Examples

### Example 1: Visual Odometry (Monocular) - Concept

```cpp
/**
 * @brief Simple Monocular VO Step
 * @param img1 Previous Frame
 * @param img2 Current Frame
 * @param K Camera Intrinsic Matrix
 * @param R, t Output Rotation and Translation
 */
void estimate_motion(cv::Mat& img1, cv::Mat& img2, cv::Mat& K, cv::Mat& R, cv::Mat& t) {
    // 1. Detect Features
    std::vector<cv::Point2f> pts1, pts2;
    cv::goodFeaturesToTrack(img1, pts1, 100, 0.3, 7);
    
    // 2. Track Features (Optical Flow)
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err);
    
    // Filter good points
    std::vector<cv::Point2f> good_pts1, good_pts2;
    for(size_t i=0; i<status.size(); i++) {
        if(status[i]) {
            good_pts1.push_back(pts1[i]);
            good_pts2.push_back(pts2[i]);
        }
    }
    
    // 3. Find Essential Matrix
    // E = t^x * R
    cv::Mat E = cv::findEssentialMat(good_pts1, good_pts2, K, cv::RANSAC, 0.999, 1.0);
    
    // 4. Recover Pose
    cv::recoverPose(E, good_pts1, good_pts2, K, R, t);
}
```

### Example 2: Pose Accumulation

Tracking the global path.

```cpp
cv::Mat R_f = cv::Mat::eye(3, 3, CV_64F);
cv::Mat t_f = cv::Mat::zeros(3, 1, CV_64F);

void update_trajectory(cv::Mat& R, cv::Mat& t) {
    // T_global = T_global * T_local
    // t_f = t_f + (R_f * t)
    // R_f = R * R_f
    
    t_f = t_f + (R_f * t);
    R_f = R * R_f;
    
    // Draw t_f.at<double>(0), t_f.at<double>(2) on a map (X-Z plane)
}
```

### Example 3: Bundle Adjustment (Concept)

Refining the map and poses.
*   **Input:** Set of Camera Poses $C_j$ and 3D Points $X_i$.
*   **Objective:** Minimize Reprojection Error.
    *   $\sum_{i,j} || p_{ij} - Project(C_j, X_i) ||^2$
*   **Solver:** Levenberg-Marquardt (using g2o or Ceres).

---

## 🔬 Hands-On Lab Exercises

### Lab 1: KITTI Odometry Benchmark

**Objective:** Run your VO on standard data.

**Steps:**
1.  Download KITTI Odometry Sequence 00 (Grayscale).
2.  Run the VO code (Example 1+2) on the sequence.
3.  **Plot:** The estimated X-Z path vs Ground Truth.
4.  **Observation:** It starts well but drifts after a few hundred frames.

### Lab 2: Scale Drift (Monocular)

**Objective:** Understand the scale ambiguity.

**Steps:**
1.  Run Monocular VO.
2.  **Issue:** Monocular vision cannot measure absolute scale (Is the car 1 meter away moving 1 m/s, or 10 meters away moving 10 m/s?).
3.  **Observation:** The trajectory shape is correct, but the size is arbitrary.
4.  **Fix:** Use Stereo Camera (Day 46) or IMU fusion.

### Lab 3: Loop Closure Detection (Bag of Words)

**Objective:** Recognize a previous image.

**Steps:**
1.  Extract ORB features from Image A.
2.  Convert descriptors to a "Bag of Words" vector (histogram of visual words).
3.  Compare with Image B's vector.
4.  **Result:** High similarity score indicates a potential loop closure.

---

## 🐛 Debugging Techniques

### Debug 1: Tracking Lost

**Symptom:** System resets or freezes.

**Cause:**
*   Fast motion (blur).
*   Textureless area (white wall).
*   **Fix:** Use a wider FoV camera (Fish-eye) or add an IMU (VIO - Visual Inertial Odometry).

### Debug 2: "Kidnapped Robot" Problem

**Symptom:** Camera is moved manually to a new location. System is lost.

**Cause:**
*   Current view doesn't match the last known location.
*   **Fix:** Global Relocalization. Search the entire map database for a match (Bag of Words).

---

## ⚡ Performance Optimization

### Optimization 1: Keyframe Selection

*   Don't add every frame to the map.
*   Only add a "Keyframe" when the camera has moved significantly or many features are lost.
*   Reduces the size of the Bundle Adjustment problem.

### Optimization 2: Local Bundle Adjustment

*   Only optimize the last N keyframes (Local Window).
*   Fix the older keyframes (Fixed Window).
*   Keeps computation constant O(1) regardless of mission duration.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does Monocular SLAM suffer from Scale Drift?**
2.  **What is the difference between "Sparse" and "Dense" SLAM?**
3.  **How does "Bundle Adjustment" refine the map?**
4.  **Why is Loop Closure essential for long-term mapping?**

### Practical Challenges

1.  **Implement "Map Saving":** Save the 3D points and Camera Poses to a file. Load them later to visualize the path.
2.  **Integrate IMU:** Use the accelerometer to estimate the scale of translation (Metric Scale).

---

## 📚 Further Reading & Resources

### Papers
*   **"ORB-SLAM: A Versatile and Accurate Monocular SLAM System"** - Mur-Artal et al.
*   **"PTAM: Parallel Tracking and Mapping"** - Klein & Murray.

### Datasets
*   **KITTI Vision Benchmark Suite.**
*   **EuRoC MAV Dataset.**

---

## 🎓 Summary

Today we covered:
- ✅ **SLAM:** Localization + Mapping.
- ✅ **VO:** Frame-to-frame motion estimation.
- ✅ **Drift:** The enemy of dead reckoning.
- ✅ **Loop Closure:** The cure for drift.
- ✅ **Optimization:** Bundle Adjustment.

**Next:** Day 49 - Structure from Motion (SfM) & Photogrammetry.

---

**Day 48 Complete** | Phase 3: Camera Systems & ISP | Week 8: Depth Sensing & 3D Vision
