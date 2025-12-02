# Day 148: Implementing OpenCV Calibration
## Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision

---

## 🎯 Learning Objectives
1.  **Develop** a robust Python script for camera calibration.
2.  **Process** a dataset of checkerboard images to extract corners.
3.  **Compute** the Camera Matrix ($K$) and Distortion Coefficients ($D$).
4.  **Evaluate** the calibration quality using Reprojection Error.
5.  **Undistort** images using the computed parameters.
6.  **Save** and **Load** calibration data using YAML/JSON.

---

## 📚 Prerequisites & Preparation
*   **Dataset:** The 20 images captured in Day 147 Lab 1.
*   **Software:** Python, OpenCV, Glob, PyYAML.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Calibration Pipeline
1.  **Object Points:** Define the 3D coordinates of the board corners in the real world (usually $Z=0$). E.g., $(0,0,0), (1,0,0), (2,0,0)...$
2.  **Image Points:** Find the 2D pixel coordinates of the corners in the image.
3.  **Solve:** Use `cv2.calibrateCamera` to find the parameters that minimize the distance between the projected Object Points and the measured Image Points.

### 🔹 Part 2: Reprojection Error
*   The ultimate metric of success.
*   For every corner, we take the 3D Object Point, project it using our new $K$ and $D$, and compare it to the detected 2D Image Point.
*   **RMS (Root Mean Square) Error:** Should be $< 0.5$ pixels for high-quality vision, $< 1.0$ for general use.

---

## 💻 Implementation Examples

### Example 1: The Calibration Script (`calibrate.py`)

```python
import numpy as np
import cv2
import glob
import yaml

# Settings
CHECKERBOARD = (9, 6) # Internal corners (Rows, Cols)
SQUARE_SIZE = 25.0    # Millimeters

# Arrays to store object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points: (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Load Images
images = glob.glob('calibration_images/*.jpg')
print(f"Found {len(images)} images.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        objpoints.append(objp)
        
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display (optional)
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)
    else:
        print(f"Warning: Corners not found in {fname}")

cv2.destroyAllWindows()

# Calibrate
print("Calibrating...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(f"Reprojection Error: {ret:.4f} pixels")
print("Camera Matrix:\n", mtx)
print("Distortion:\n", dist)

# Save
data = {
    'camera_matrix': mtx.tolist(),
    'dist_coeff': dist.tolist(),
    'reprojection_error': ret
}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)
print("Saved to calibration.yaml")
```

### Example 2: Undistortion Script (`undistort.py`)

Using the saved parameters.

```python
import cv2
import yaml
import numpy as np

# Load
with open("calibration.yaml", "r") as f:
    data = yaml.safe_load(f)

mtx = np.array(data['camera_matrix'])
dist = np.array(data['dist_coeff'])

# Read Image
img = cv2.imread('test_image.jpg')
h, w = img.shape[:2]

# Optimization: Get Optimal New Camera Matrix
# This crops the image to remove black borders (alpha=0) or keeps them (alpha=1)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Crop
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv2.imshow('Original', img)
cv2.imshow('Undistorted', dst)
cv2.waitKey(0)
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Run the Calibration

**Objective:** Get your own $K$ and $D$.

**Steps:**
1.  Put your 20 images in a folder.
2.  Measure your square size (e.g., 24mm). Update the script.
3.  Run `calibrate.py`.
4.  **Goal:** Achieve Error < 0.5.
5.  **Troubleshoot:** If Error > 1.0, look at the output. Did it fail to detect corners in some images? Remove blurry images and retry.

### Lab 2: Compare Fisheye vs Standard

**Objective:** See the difference in $D$.

**Steps:**
1.  Calibrate a standard webcam (Low distortion). $k_1$ will be small (e.g., 0.05).
2.  Calibrate a wide-angle/fisheye lens. $k_1$ will be large and negative (e.g., -0.3).
3.  **Visual Check:** Undistort the fisheye image. Straight lines should become straight.

### Lab 3: Live Undistortion

**Objective:** Real-time correction.

**Steps:**
1.  Modify the `undistort.py` to open `cv2.VideoCapture(0)`.
2.  Apply `cv2.undistort` on every frame.
3.  **Performance:** Measure FPS. Undistortion is computationally expensive on CPU.
4.  **Optimization:** Use `cv2.initUndistortRectifyMap` once, then `cv2.remap` in the loop. It's faster.

---

## 🐛 Debugging Calibration Results

### Debug 1: "Zoomed In" Undistortion

**Symptom:** The undistorted image looks zoomed in and cropped.

**Cause:**
*   `getOptimalNewCameraMatrix` with `alpha=0`.
*   **Fix:** Set `alpha=1` to keep all pixels (but you get black curved borders).

### Debug 2: Wobbly Lines

**Symptom:** Straight lines look wavy (mustache distortion).

**Cause:**
*   Higher order coefficients ($k_3$) are unstable if not enough data at the corners.
*   **Fix:** Set `CALIB_FIX_K3` flag in `calibrateCamera` to force $k_3=0$.

---

## ⚡ Performance Optimization

### Optimization 1: Remapping

*   `cv2.undistort` calculates the map every time.
*   `cv2.initUndistortRectifyMap` pre-calculates the lookup table (LUT).
*   `cv2.remap` just applies the LUT.
*   **Speedup:** 2x-3x.

### Optimization 2: GPU Undistortion

*   Use `cv2.cuda.remap` or OpenGL shaders.
*   Essential for 4K video.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the unit of the Focal Length in the Camera Matrix?** (Pixels).
2.  **Why do we need at least 3 images to calibrate?** (Each image provides constraints. In practice, 10-20 are needed for noise reduction).
3.  **What is "Skew" ($s$) in the Camera Matrix?** (Usually 0. It means the pixels are not rectangular/grid is not orthogonal. Rare in modern sensors).

### Practical Challenges

1.  **Calibrate a Phone:** Use your smartphone. Transfer images to PC. Calibrate. Note the focal length in pixels. Compare with EXIF data (Focal Length in mm) using the sensor pixel size.
2.  **Stereo Prep:** If you have two cameras, calibrate them *individually* first. Save `left.yaml` and `right.yaml`.

---

## 📚 Further Reading & Resources

### Documentation
*   **OpenCV `calibrateCamera` flags.**
*   **Aruco Module for Charuco calibration.**

---

## 🎓 Summary

Today we covered:
- ✅ **Pipeline:** Object Points -> Image Points -> Solve.
- ✅ **Code:** Python script for batch processing.
- ✅ **Metrics:** Reprojection Error.
- ✅ **Undistortion:** Making lines straight.
- ✅ **Optimization:** Using Remap.

**Next:** Day 149 - Stereo Vision Fundamentals.

---

**Day 148 Complete** | Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision


