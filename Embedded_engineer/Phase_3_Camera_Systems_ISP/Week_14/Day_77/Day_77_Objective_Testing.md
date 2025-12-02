# Day 77: Objective Testing with Imatest & Open Source Tools
## Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Understand** the role of Imatest in the camera industry (The "Gold Standard").
2.  **Implement** Color Accuracy testing using Python (`colour-science`) and Macbeth Charts.
3.  **Calculate** Color Error ($\Delta E_{76}$ and $\Delta E_{2000}$).
4.  **Measure** Lens Distortion (TV Distortion) objectively.
5.  **Automate** the testing pipeline: Capture -> Analyze -> Report.
6.  **Compare** ISP Tuning iterations using objective data.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Macbeth ColorChecker (24 patches).
*   **Software:** Python (`pip install colour-science opencv-python matplotlib`), Imatest (Optional/Trial).
*   **Knowledge:** CIE Lab Color Space.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Imatest Overview
*   **What is it?** A software suite for analyzing image quality.
*   **Modules:**
    *   **SFRplus:** Sharpness, Distortion, FOV from one chart.
    *   **Colorcheck:** Color accuracy, Noise, White Balance.
    *   **Stepchart:** Dynamic Range.
*   **Why use it?** Standardized results that suppliers and OEMs agree on.

### 🔹 Part 2: Color Accuracy ($\Delta E$)
*   **RGB is not perceptual.** A difference of (10,0,0) in RGB might look different than (0,10,0).
*   **Lab Color Space:** Designed to be perceptually uniform.
    *   **L:** Lightness.
    *   **a:** Green-Red axis.
    *   **b:** Blue-Yellow axis.
*   **$\Delta E$ (Delta E):** The Euclidean distance between two colors in Lab space.
    *   $\Delta E < 2$: Indistinguishable to human eye.
    *   $\Delta E > 10$: Obvious wrong color.

### 🔹 Part 3: Lens Distortion
*   **Barrel Distortion:** Straight lines bow outwards (Wide angle).
*   **Pincushion Distortion:** Straight lines bow inwards (Telephoto).
*   **SMIA TV Distortion:** A standard metric to quantify the bowing as a percentage (%).

---

## 💻 Implementation Examples

### Example 1: Color Accuracy Analysis (Python)

Calculating $\Delta E$ for a Macbeth Chart.

```python
import cv2
import numpy as np
import colour

# 1. Reference Values (Macbeth ColorChecker 24 - sRGB)
# (Simplified: In reality, convert Lab reference to sRGB)
refs_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(refs_srgb))

# 2. Extract Patches from Image
def extract_patches(image, corners):
    # Perspective Transform to rectify chart
    # Grid sampling to get mean RGB of each patch
    return measured_rgb

# 3. Convert Measured RGB to Lab
measured_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(measured_rgb))

# 4. Calculate Delta E
delta_e = colour.delta_E(refs_lab, measured_lab, method='CIE 2000')

print(f"Mean Delta E: {np.mean(delta_e):.2f}")
print(f"Max Delta E: {np.max(delta_e):.2f}")
```

### Example 2: Distortion Measurement

Using OpenCV calibration to find distortion coefficients ($k_1, k_2, k_3$).

```python
import cv2
import glob

# Prepare object points (0,0,0), (1,0,0), ...
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibration_*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# Calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(f"Distortion Coefficients (k1, k2, p1, p2, k3): {dist}")
```

### Example 3: Automated Test Script

```bash
#!/bin/bash
# run_iq_test.sh

# 1. Capture Image
v4l2-ctl --stream-mmap --stream-count=1 --stream-to=test.raw

# 2. Convert to TIFF (using dcraw or custom tool)
dcraw -T test.raw

# 3. Run Analysis
python3 analyze_color.py test.tiff > report.txt
python3 analyze_sharpness.py test.tiff >> report.txt

# 4. Check Pass/Fail
grep "FAIL" report.txt && exit 1 || exit 0
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Tuning Color Matrix (CCM)

**Objective:** Minimize $\Delta E$.

**Steps:**
1.  Capture image of Macbeth Chart.
2.  Run Analysis (Example 1). Note Mean $\Delta E$ (e.g., 15.0).
3.  Adjust ISP CCM (Color Correction Matrix).
    *   If Red is too weak, boost $RR$ component.
    *   Or use a solver (Least Squares) to compute the ideal CCM from the measured RGB vs Reference RGB.
4.  Apply new CCM.
5.  Capture and Analyze again.
6.  **Goal:** Mean $\Delta E < 5$.

### Lab 2: White Balance Verification

**Objective:** Check Gray Neutrality.

**Steps:**
1.  Analyze the bottom row of the Macbeth Chart (6 Gray patches).
2.  In Lab space, $a$ and $b$ should be 0 for neutral gray.
3.  **Metric:** $\Delta C = \sqrt{a^2 + b^2}$ (Chroma error).
4.  **Goal:** $\Delta C < 2$ for all gray patches.

### Lab 3: Distortion Check

**Objective:** Measure lens quality.

**Steps:**
1.  Capture a checkerboard.
2.  Run OpenCV calibration.
3.  **Visualize:** Use `cv2.undistort` to correct the image.
4.  **Compare:** Toggle between Distorted and Undistorted.
5.  **Observation:** Fisheye lenses have massive $k_1$ (negative).

---

## 🐛 Debugging Validation Issues

### Debug 1: High Delta E on Red

**Symptom:** Reds look orange or desaturated.

**Cause:**
*   IR Cut Filter is leaking IR light (Red + IR = washed out).
*   Crosstalk (Green pixels sensitive to Red light).
*   **Fix:** Better IR Cut Filter or aggressive CCM (which increases noise).

### Debug 2: Inconsistent Results

**Symptom:** Run test twice, get different results.

**Cause:**
*   Lighting flicker (50Hz/60Hz).
*   Auto-Exposure settling time.
*   **Fix:** Use DC lighting. Discard first 10 frames. Lock AE before capture.

---

## ⚡ Performance Optimization

### Optimization 1: Region of Interest (ROI)

*   Don't analyze the whole 4K image if you only care about the center chart.
*   Crop the chart area first. Speeds up processing by 10x.

### Optimization 2: Parallel Testing

*   Run Color Analysis and Sharpness Analysis in parallel threads.
*   Use GPU for image conversion (OpenCV CUDA).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is $\Delta E_{2000}$ better than $\Delta E_{76}$?** (It corrects for human eye sensitivity differences in saturation and hue).
2.  **What is "Shading Correction" (LSC)?** (Fixing the brightness drop-off at corners).
3.  **How does "Distortion" affect Object Detection?** (Straight lines become curved, confusing the AI model).
4.  **What is a "Golden Sample"?** (A perfect camera unit used as a reference for production testing).

### Practical Challenges

1.  **Build a "Color Solver":** Write a script that takes Measured RGB and Reference RGB, and outputs the $3 \times 3$ CCM that minimizes error.
2.  **Create a HTML Report:** Generate a nice report with graphs (Matplotlib) showing SNR vs Gain and Color Error vectors.

---

## 📚 Further Reading & Resources

### Documentation
*   **Colour-Science Library Documentation.**
*   **Imatest Documentation (Great theory resource even if you don't buy it).**

---

## 🎓 Summary

Today we covered:
- ✅ **Imatest:** The industry standard.
- ✅ **Color:** Lab space, Delta E.
- ✅ **Distortion:** Calibration coefficients.
- ✅ **Automation:** Scripting the lab.
- ✅ **Tuning:** Using data to improve the ISP.

**Next:** Day 78 - Color Accuracy & White Balance Verification (Deep Dive).

---

**Day 77 Complete** | Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance
