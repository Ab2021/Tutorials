# Day 50: Week 8 Review - 3D Scanning Project
## Phase 3: Camera Systems & ISP | Week 8: Depth Sensing & 3D Vision

---

## 🎯 Learning Objectives
1.  **Integrate** Stereo Calibration, Disparity Computation, and Point Cloud Generation into a single pipeline.
2.  **Build** a "3D Scanner" application that captures a scene and exports a `.ply` file.
3.  **Implement** post-processing filters (Outlier Removal, Smoothing) to improve scan quality.
4.  **Validate** the accuracy of the scanner using ground truth measurements.
5.  **Visualize** the result in real-time (optional, using PCL Visualizer).

---

## 📚 Week 8 Recap

### Topics Covered

**Day 46: Stereo Vision**
- Epipolar Geometry, Rectification.
- Disparity Calculation ($Z = fB/d$).
- Block Matching (BM) vs SGM.

**Day 47: 3D Reconstruction**
- Point Clouds (XYZRGB).
- PCL Library (Filtering, Downsampling).
- Meshing.

**Day 48: SLAM**
- Localization + Mapping.
- Visual Odometry (VO).
- Loop Closure.

**Day 49: SfM & Photogrammetry**
- Offline 3D reconstruction from unordered photos.
- COLMAP pipeline.

---

## 💻 Week 8 Integration Project

### Project: "StereoMapper - A Real-Time 3D Scanner"

**Objective:** Build a C++ application that connects to a stereo camera (or two webcams), computes the depth map, converts it to a point cloud, filters noise, and saves the result.

**Features:**
1.  **Capture:** Dual-stream capture (Left/Right).
2.  **Rectify:** Apply calibration maps.
3.  **Compute:** StereoBM / SGM.
4.  **Filter:** Statistical Outlier Removal.
5.  **Export:** Save as PLY/PCD.

### Architecture

```mermaid
graph LR
    CAM[Stereo Camera] --> RECT[Rectification]
    RECT --> DISP[Disparity (SGM)]
    DISP --> PROJ[Reproject to 3D]
    PROJ --> FILT[Outlier Filter]
    FILT --> SAVE[Save .PLY]
    FILT --> VIS[Visualizer]
```

### Implementation

#### Part 1: The Scanner Class

```cpp
class StereoScanner {
    cv::Mat M1, D1, M2, D2, R, T; // Calibration
    cv::Mat map11, map12, map21, map22; // Rectification Maps
    cv::Mat Q; // Reprojection Matrix
    cv::Ptr<cv::StereoSGBM> matcher;
    
public:
    StereoScanner(std::string calib_file) {
        load_calibration(calib_file);
        
        // Init Rectification Maps
        cv::Mat R1, R2, P1, P2;
        cv::stereoRectify(M1, D1, M2, D2, cv::Size(1280, 720), R, T, R1, R2, P1, P2, Q);
        cv::initUndistortRectifyMap(M1, D1, R1, P1, cv::Size(1280, 720), CV_16SC2, map11, map12);
        cv::initUndistortRectifyMap(M2, D2, R2, P2, cv::Size(1280, 720), CV_16SC2, map21, map22);
        
        // Init Matcher (SGM)
        matcher = cv::StereoSGBM::create(0, 128, 3);
        matcher->setBlockSize(5);
        matcher->setP1(8 * 3 * 5 * 5);
        matcher->setP2(32 * 3 * 5 * 5);
        matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    }
    
    void process_frame(cv::Mat& left_raw, cv::Mat& right_raw) {
        cv::Mat left_rect, right_rect;
        cv::remap(left_raw, left_rect, map11, map12, cv::INTER_LINEAR);
        cv::remap(right_raw, right_rect, map21, map22, cv::INTER_LINEAR);
        
        cv::Mat disp, disp_vis;
        matcher->compute(left_rect, right_rect, disp);
        
        // Convert to PCL
        auto cloud = disparity_to_pcl(disp, left_rect, Q);
        
        // Filter
        auto cloud_filtered = filter_cloud(cloud);
        
        // Save
        pcl::io::savePLYFileBinary("scan.ply", *cloud_filtered);
    }
};
```

#### Part 2: Calibration Utility

Before running the scanner, you MUST calibrate.

```cpp
void run_calibration() {
    // 1. Capture 20 pairs of Checkerboard images.
    // 2. Find corners (findChessboardCorners).
    // 3. Run stereoCalibrate.
    // 4. Save matrices to XML/YAML.
    
    double rms = cv::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, 
                                     M1, D1, M2, D2, imageSize, R, T, E, F);
    std::cout << "Calibration RMS Error: " << rms << std::endl;
}
```

---

## 🔬 System Validation Plan

### Test 1: Flat Wall Test
**Objective:** Verify Rectification and Disparity flatness.
**Procedure:**
1.  Point camera at a flat textured wall.
2.  Compute Disparity.
3.  **Goal:** Disparity should be constant (flat plane) across the image.
4.  **Failure:** If disparity curves or tilts, calibration is wrong.

### Test 2: Z-Accuracy Test
**Objective:** Measure depth error.
**Procedure:**
1.  Place object at 1.0m (measured with tape).
2.  Scanner measures Z.
3.  Place object at 2.0m.
4.  Scanner measures Z.
5.  **Goal:** Error < 2% (e.g., +/- 2cm at 1m).

### Test 3: Point Cloud Density
**Objective:** Ensure good coverage.
**Procedure:**
1.  Scan a complex object (e.g., a chair).
2.  Check for holes.
3.  **Fix:** Adjust SGM parameters (`P1`, `P2`, `uniquenessRatio`) to fill gaps.

---

## 🐛 Troubleshooting Guide

### Issue 1: "Staircase" Artifacts in Depth

**Symptom:** Depth looks like discrete layers.

**Cause:**
*   Sub-pixel interpolation disabled.
*   Disparity resolution too low.
*   **Fix:** Use SGM (which does sub-pixel estimation). Increase resolution.

### Issue 2: Left/Right Swapped

**Symptom:** Background is close, Foreground is far (Inverted depth).

**Cause:**
*   Cameras plugged in wrong ports.
*   **Fix:** Swap `imgL` and `imgR` in code.

---

## 📝 Assessment Questions

### Comprehensive Questions

1.  **Why does the error in $Z$ increase with distance?** (Derive $\Delta Z$ from $\Delta d$).
2.  **What is the role of `P1` and `P2` in Semi-Global Matching?**
3.  **How does "Active Stereo" (Projector) improve results?**
4.  **Explain the difference between "Intrinsic" and "Extrinsic" parameters.**

### Practical Challenges

1.  **Implement "Color-Coded Depth":** Map near objects to Red, far objects to Blue (Jet colormap) for visualization.
2.  **Build a "Volume Estimator":** Scan a box. Calculate its bounding box volume ($W \times H \times D$).

---

## 📚 Resources & Next Steps

### Week 8 Summary

**Completed:**
- ✅ **Stereo:** The math of two eyes.
- ✅ **3D:** Point clouds and meshes.
- ✅ **SLAM:** Mapping the world.
- ✅ **Project:** Building a scanner.

**Key Skills Acquired:**
- Camera Calibration.
- 3D Data Processing.
- Depth Estimation.

### Week 9 Preview (Phase 3 Continued)

**Topics:**
- **Video Encoding:** H.264/H.265, Bitrate Control.
- **Streaming:** RTSP, WebRTC.
- **Latency Optimization:** Glass-to-Glass < 100ms.
- **GStreamer Deep Dive.**

---

**Day 50 Complete** | Phase 3: Camera Systems & ISP | Week 8: Depth Sensing & 3D Vision
