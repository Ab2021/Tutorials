# Day 41: Feature Detection & Tracking
## Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI

---

## 🎯 Learning Objectives
1.  **Understand** what makes a "Feature" (Corner, Blob, Edge).
2.  **Implement** Corner Detection: Harris Corner Detector.
3.  **Use** Fast Feature Detectors: FAST and ORB (Oriented FAST and Rotated BRIEF).
4.  **Perform** Feature Matching: Brute-Force and FLANN.
5.  **Track** Objects: Sparse Optical Flow (Lucas-Kanade).
6.  **Debug** tracking failures (Occlusion, Lighting changes).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera input.
*   **Software:** OpenCV.
*   **Knowledge:** Gradients, Taylor Series (for Optical Flow derivation).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is a Feature?
*   **Flat Region:** No gradient in any direction. Bad for tracking.
*   **Edge:** Gradient in one direction. Good for tracking perpendicular motion (Aperture Problem).
*   **Corner:** Gradient in two directions. Unique and trackable.

### 🔹 Part 2: Detectors
*   **Harris Corner:** Computes the structure tensor (autocorrelation matrix). If both eigenvalues are large, it's a corner. Robust but slow.
*   **FAST (Features from Accelerated Segment Test):** Checks a circle of 16 pixels around a candidate. If N contiguous pixels are brighter/darker than center, it's a corner. Very fast.
*   **ORB (Oriented FAST and Rotated BRIEF):** A free alternative to SIFT/SURF. Adds orientation to FAST and uses binary descriptors (BRIEF) for matching.

### 🔹 Part 3: Optical Flow (Lucas-Kanade)
*   **Assumption:** Pixel intensity doesn't change between frames ($I(x,y,t) = I(x+dx, y+dy, t+dt)$). Motion is small.
*   **Method:** Solves a system of linear equations for a small window (e.g., 3x3) to find $(u, v)$ velocity vector.
*   **Pyramids:** Used to handle large motions (Coarse-to-Fine).

---

## 💻 Implementation Examples

### Example 1: Harris Corner Detection

```cpp
/**
 * @brief Harris Corner Demo
 */
void detect_harris(cv::Mat& img) {
    cv::Mat gray, dst;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    // Block size 2, Aperture 3, k=0.04
    cv::cornerHarris(gray, dst, 2, 3, 0.04);
    
    // Normalize and Threshold
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            if ((int)dst.at<float>(y,x) > 200) {
                cv::circle(img, cv::Point(x,y), 5, cv::Scalar(0,0,255), 2);
            }
        }
    }
}
```

### Example 2: ORB Feature Matching

Matching features between two images.

```cpp
void match_orb(cv::Mat& img1, cv::Mat& img2) {
    // 1. Detect and Compute
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    
    detector->detectAndCompute(img1, cv::noArray(), kp1, des1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, des2);
    
    // 2. Match (Brute Force Hamming)
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(des1, des2, matches);
    
    // 3. Sort and Draw Top 10
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + 10);
    
    cv::Mat img_matches;
    cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches);
    
    cv::imshow("Matches", img_matches);
}
```

### Example 3: Optical Flow (Lucas-Kanade)

Tracking points from Frame N to Frame N+1.

```cpp
void track_optical_flow() {
    cv::VideoCapture cap(0);
    cv::Mat old_frame, old_gray;
    std::vector<cv::Point2f> p0, p1;
    
    // Initialize
    cap >> old_frame;
    cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
    
    // Detect initial points to track
    cv::goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7);
    
    while(true) {
        cv::Mat frame, frame_gray;
        cap >> frame;
        if (frame.empty()) break;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        
        // Calculate Flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err);
        
        // Select good points
        std::vector<cv::Point2f> good_new;
        for (size_t i = 0; i < p0.size(); i++) {
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                cv::line(frame, p0[i], p1[i], cv::Scalar(0,255,0), 2);
                cv::circle(frame, p1[i], 5, cv::Scalar(0,255,0), -1);
            }
        }
        
        cv::imshow("Tracking", frame);
        
        // Update
        old_gray = frame_gray.clone();
        p0 = good_new;
        
        // Re-detect if points lost
        if (p0.size() < 10) {
             cv::goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7);
        }
        
        if (cv::waitKey(30) == 27) break;
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: FAST vs Harris Speed Test

**Objective:** Benchmark detectors.

**Steps:**
1.  Load a video stream.
2.  Measure FPS running Harris.
3.  Measure FPS running FAST.
4.  **Observation:** FAST should be significantly faster (suitable for real-time on Raspberry Pi), but might detect more noise.

### Lab 2: Panorama Stitching (Homography)

**Objective:** Stitch two overlapping images.

**Steps:**
1.  Detect ORB features in Image A and Image B.
2.  Match features.
3.  Find Homography Matrix H using RANSAC (rejects outliers).
4.  Warp Image B using H to align with Image A.
5.  Blend them.

### Lab 3: Stabilization

**Objective:** Remove camera shake.

**Steps:**
1.  Track features between consecutive frames.
2.  Calculate the average motion vector (dx, dy).
3.  Shift the current frame by (-dx, -dy) to cancel the motion.
4.  **Result:** Smooth video.

---

## 🐛 Debugging Techniques

### Debug 1: Tracking Drift

**Symptom:** The tracking point slowly slides off the object.

**Cause:**
*   Accumulation of small errors in Optical Flow.
*   Appearance change (rotation/scale) not modeled by simple translation.
*   **Fix:** Use "Template Matching" or re-initialize features periodically.

### Debug 2: Feature Starvation

**Symptom:** Tracking stops in low texture areas (white wall).

**Cause:**
*   `goodFeaturesToTrack` returns 0 points.
*   **Fix:** Lower the "Quality Level" threshold. Or switch to edge tracking if corners are missing.

---

## ⚡ Performance Optimization

### Optimization 1: Pyramidal LK on GPU

*   Optical Flow is parallelizable.
*   Use `cv::cuda::SparsePyrLKOpticalFlow` on Jetson Nano.
*   Speedup: 10x-50x.

### Optimization 2: Grid-Based Detection

*   Instead of detecting 100 features globally (which might cluster in one corner), divide the image into a 5x5 grid.
*   Detect 4 features in each grid cell.
*   Ensures uniform coverage for better motion estimation.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why are corners better features than edges?**
2.  **What is the "Aperture Problem" in Optical Flow?**
3.  **How does RANSAC help in feature matching?**
4.  **Why do we need Binary Descriptors (like BRIEF) for embedded systems?**

### Practical Challenges

1.  **Implement "Visual Odometry":** Estimate the path of the camera by integrating the motion vectors over time.
2.  **Create a "Virtual Button":** Define a region in the air. If Optical Flow detects motion in that region (hand wave), trigger an action.

---

## 📚 Further Reading & Resources

### Papers
*   **"A Combined Corner and Edge Detector"** - Harris & Stephens.
*   **"ORB: An efficient alternative to SIFT or SURF"** - Rublee et al.

---

## 🎓 Summary

Today we covered:
- ✅ **Features:** Corners are king.
- ✅ **Detectors:** Harris (Accurate) vs FAST (Speed).
- ✅ **Descriptors:** ORB for matching.
- ✅ **Tracking:** Lucas-Kanade Optical Flow.
- ✅ **Applications:** Stabilization and Stitching.

**Next:** Day 42 - Object Detection with ML (Haar Cascades & HOG).

---

**Day 41 Complete** | Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI
