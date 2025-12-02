# Day 40: Computer Vision Basics (OpenCV)
## Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI

---

## 🎯 Learning Objectives
1.  **Install** and configure OpenCV (C++ and Python) for embedded systems.
2.  **Understand** the `cv::Mat` data structure and memory management.
3.  **Perform** basic image operations: Reading, Writing, Resizing, Cropping.
4.  **Analyze** Color Spaces: RGB, HSV (for color tracking), YUV, GRAY.
5.  **Apply** Image Filtering: Gaussian Blur, Median Blur, Morphological Operations (Erode/Dilate).
6.  **Debug** common OpenCV errors (Empty matrix, Type mismatch).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** PC or Embedded Board (Raspberry Pi).
*   **Software:** OpenCV 4.x (`sudo apt install libopencv-dev python3-opencv`).
*   **Knowledge:** Matrix Algebra.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The `cv::Mat` Structure
*   **Header:** Contains metadata (width, height, type, pointer to data). Small size.
*   **Data:** The actual pixel array. Large size.
*   **Reference Counting:** `cv::Mat B = A` does NOT copy the data. It creates a new header pointing to the same data. To copy, use `A.clone()`.

### 🔹 Part 2: Color Spaces for Vision
*   **BGR:** Default in OpenCV. Not good for color segmentation because Lightness is mixed with Color.
*   **HSV (Hue, Saturation, Value):**
    *   **Hue:** The color type (0-179 in OpenCV). Robust to lighting changes.
    *   **Saturation:** How "pure" the color is.
    *   **Value:** Brightness.
    *   *Usage:* Tracking a red ball is easy in HSV (Hue range 0-10 and 170-180), hard in RGB.
*   **YCrCb / YUV:** Used for skin detection and video compression.

### 🔹 Part 3: Morphological Operations
Processing binary images (masks) based on shapes.
*   **Erosion:** Shrinks white regions. Removes small noise (salt).
*   **Dilation:** Expands white regions. Fills holes (pepper).
*   **Opening:** Erosion followed by Dilation. Removes noise.
*   **Closing:** Dilation followed by Erosion. Closes holes.

---

## 💻 Implementation Examples

### Example 1: Basic OpenCV (C++)

```cpp
/**
 * @file basic_cv.cpp
 * @brief Load, Resize, Save
 */

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Read Image
    cv::Mat img = cv::imread("input.jpg");
    if (img.empty()) {
        std::cerr << "Could not open image!" << std::endl;
        return -1;
    }
    
    // 2. Resize (Downscale by 2)
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    
    // 3. Convert to Grayscale
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    
    // 4. Save
    cv::imwrite("output_gray.png", gray);
    
    return 0;
}
```

### Example 2: Color Segmentation (HSV)

Detecting a blue object.

```python
import cv2
import numpy as np

def detect_blue(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define Blue Range (OpenCV Hue is 0-179)
    # Blue is around 120 degrees (60 in 0-179 scale? No, 120 degrees is 120/2 = 60? Wait.)
    # Red=0, Green=60, Blue=120. In OpenCV (0-180), Blue is 120.
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Threshold
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Clean up noise (Opening)
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Bitwise AND to show only blue parts
    res = cv2.bitwise_and(img, img, mask=mask_clean)
    
    cv2.imshow('Result', res)
    cv2.waitKey(0)
```

### Example 3: Accessing Pixels Efficiently

Do NOT use `at<Vec3b>(y,x)` inside tight loops if performance matters. Use pointers.

```cpp
void invert_colors_fast(cv::Mat& img) {
    // Ensure continuous memory
    if (!img.isContinuous()) return;
    
    int size = img.total() * img.channels();
    uchar* ptr = img.ptr<uchar>(0);
    
    for (int i = 0; i < size; i++) {
        ptr[i] = 255 - ptr[i];
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Invisible Cloak"

**Objective:** Use HSV masking to replace a color with the background.

**Steps:**
1.  Capture a static background frame.
2.  Enter the frame holding a green cloth.
3.  **Process:**
    *   Convert current frame to HSV.
    *   Create a mask for Green.
    *   `Output = (Background * Mask) + (CurrentFrame * (1 - Mask))`.
4.  **Result:** The green cloth disappears, revealing the background behind it.

### Lab 2: Document Scanner (Perspective Transform)

**Objective:** Rectify a tilted document.

**Steps:**
1.  Detect edges (Canny).
2.  Find Contours (`findContours`).
3.  Find the largest 4-sided polygon.
4.  Apply `getPerspectiveTransform` and `warpPerspective` to map the corners to a rectangle.

### Lab 3: High Speed Filtering

**Objective:** Compare Gaussian Blur speed.

**Steps:**
1.  Load a 4K image.
2.  Measure time for `cv::GaussianBlur` with kernel 5x5.
3.  Measure time for `cv::boxFilter`.
4.  **Observation:** Box filter is much faster (O(1) with integral images) compared to Gaussian (O(K^2) or O(K)).

---

## 🐛 Debugging Techniques

### Debug 1: "Assertion Failed: size.width > 0"

**Symptom:** Crash on `imshow` or `resize`.

**Cause:**
*   Image path is wrong (file not found). `imread` returns empty matrix, but doesn't throw exception.
*   **Fix:** Always check `if (img.empty())` after loading.

### Debug 2: Weird Colors (Blue Skin)

**Symptom:** People look like Avatars.

**Cause:**
*   OpenCV uses **BGR** order. Matplotlib/Qt use **RGB**.
*   **Fix:** Use `cv::cvtColor(img, out, cv::COLOR_BGR2RGB)` before displaying in non-OpenCV windows.

---

## ⚡ Performance Optimization

### Optimization 1: Region of Interest (ROI)

*   Don't process the whole image if you only care about the center.
*   `cv::Mat roi = img(cv::Rect(x, y, w, h));`
*   This is O(1) (no copy). Processing `roi` is faster.

### Optimization 2: Parallel Loops (`cv::parallel_for_`)

*   OpenCV has a built-in parallel framework.
*   Use it instead of raw `std::thread` for pixel-wise operations.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is HSV better than RGB for color detection?**
2.  **What is the difference between `clone()` and `=` operator for `cv::Mat`?**
3.  **How does "Erosion" remove noise?**
4.  **Why is the Canny Edge Detector considered "optimal"?**

### Practical Challenges

1.  **Implement a "Skin Detector":** Use YCrCb range (Cr: 133-173, Cb: 77-127) to mask skin pixels.
2.  **Create a "Vignette Filter":** Multiply the image by a radial gradient mask to darken corners.

---

## 📚 Further Reading & Resources

### Documentation
*   **docs.opencv.org:** The official reference.

### Books
*   **"Learning OpenCV 4"** - Bradski & Kaehler.

---

## 🎓 Summary

Today we covered:
- ✅ **OpenCV:** The standard library for CV.
- ✅ **Mat:** Memory management and pointers.
- ✅ **Color Spaces:** BGR vs HSV.
- ✅ **Morphology:** Cleaning up binary masks.
- ✅ **Optimization:** ROI and Pointers.

**Next:** Day 41 - Feature Detection & Tracking.

---

**Day 40 Complete** | Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI
