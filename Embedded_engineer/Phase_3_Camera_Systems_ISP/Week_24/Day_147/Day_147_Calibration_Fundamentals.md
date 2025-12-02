# Day 147: Camera Calibration Fundamentals
## Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision

---

## 🎯 Learning Objectives
1.  **Understand** the Pinhole Camera Model and Lens Distortion.
2.  **Define** Intrinsic Parameters (Focal Length, Principal Point) and Extrinsic Parameters (Rotation, Translation).
3.  **Analyze** Radial and Tangential Distortion coefficients ($k_1, k_2, p_1, p_2$).
4.  **Perform** a manual calculation of FOV based on sensor size and focal length.
5.  **Prepare** a Calibration Target (Checkerboard or Charuco).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with a wide-angle lens (fisheye preferred for distortion examples).
*   **Software:** Python, OpenCV, NumPy.
*   **Tools:** Printed Checkerboard (A4 or A3 size).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Pinhole Camera Model
*   **Concept:** Light passes through a tiny hole (aperture) and projects an inverted image on the sensor.
*   **Matrix Form:**
    $$
    s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
    $$
    *   **Intrinsic Matrix ($K$):** Internal properties ($f_x, f_y, c_x, c_y$).
    *   **Extrinsic Matrix ($[R|t]$):** Position of camera in the world.

### 🔹 Part 2: Lens Distortion
*   Real lenses are not perfect pinholes.
*   **Radial Distortion:** Light bends more at the edges.
    *   *Barrel:* Lines bow out (Wide angle).
    *   *Pincushion:* Lines bow in (Telephoto).
    *   Formula: $x_{distorted} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$.
*   **Tangential Distortion:** Lens is not perfectly parallel to the sensor.
    *   Formula: $x_{distorted} = x + [2p_1 xy + p_2(r^2 + 2x^2)]$.

---

## 💻 Implementation Examples

### Example 1: Generating a Calibration Board

We need a precise target.

```python
import cv2
import numpy as np

def create_board(rows=9, cols=6, square_size=50):
    # Create a white image
    width = cols * square_size
    height = rows * square_size
    board = np.ones((height, width), dtype=np.uint8) * 255
    
    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 1:
                start_x = x * square_size
                start_y = y * square_size
                cv2.rectangle(board, (start_x, start_y), 
                              (start_x + square_size, start_y + square_size), 
                              0, -1)
    return board

board = create_board()
cv2.imwrite("checkerboard.png", board)
```

### Example 2: Detecting Corners

Finding the intersection points.

```python
image = cv2.imread("capture_01.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find corners (9x6 internal corners)
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

if ret:
    # Refine corners (Sub-pixel accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Draw
    cv2.drawChessboardCorners(image, (9, 6), corners2, ret)
    cv2.imshow('Corners', image)
    cv2.waitKey(0)
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Capture Calibration Dataset

**Objective:** Get good data.

**Steps:**
1.  Print the checkerboard. Glue it to a flat board (foam core).
2.  Hold the camera steady.
3.  Capture 20 images of the board from different angles:
    *   Center, Left, Right, Top, Bottom.
    *   Tilted X, Tilted Y.
    *   Far, Near.
    *   **Crucial:** Ensure the board covers the *corners* of the image to capture distortion.

### Lab 2: Manual FOV Calculation

**Objective:** Check the spec.

**Steps:**
1.  Sensor: IMX219 (Sony).
    *   Format: 1/4 inch.
    *   Active Array: 3280 x 2464.
    *   Pixel Size: 1.12 um.
    *   Sensor Width = $3280 \times 1.12 \mu m = 3.67 mm$.
2.  Lens: Focal Length $f = 3.04 mm$.
3.  **Calculation:**
    *   $HFOV = 2 \times \arctan(\frac{SensorWidth}{2 \times f})$.
    *   $HFOV = 2 \times \arctan(\frac{3.67}{2 \times 3.04}) = 2 \times \arctan(0.603) \approx 62.2^\circ$.
4.  **Compare:** Check the datasheet.

---

## 🐛 Debugging Calibration

### Debug 1: "Corners Not Found"

**Symptom:** `findChessboardCorners` returns False.

**Cause:**
*   Lighting is uneven (shadows on the board).
*   Image is too blurry (motion blur).
*   White border around the board is too small (OpenCV needs a "Quiet Zone").
*   **Fix:** Add a white border. Use better lighting.

### Debug 2: High Reprojection Error

**Symptom:** Calibration result has error > 1.0 pixel.

**Cause:**
*   Poor quality board (paper is flexible/curved).
*   Motion blur.
*   Not enough angles.
*   **Fix:** Use a Charuco board (more robust to occlusion). Use a flat rigid target.

---

## ⚡ Performance Optimization

### Optimization 1: Charuco Boards

*   **Chessboard:** Fails if *one* corner is occluded.
*   **Charuco (Chessboard + ArUco):** Can calibrate even if part of the board is missing.
*   **Benefit:** Allows capturing corners right at the edge of the image frame without losing tracking.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the "Principal Point" ($c_x, c_y$)?** (The point where the optical axis intersects the sensor. Usually the center of the image, but not always).
2.  **Why do we need "Tangential Distortion"?** (Because the lens might be mounted slightly crooked relative to the sensor plane).
3.  **What happens if you use a calibrated camera with a different resolution?** (The Intrinsic Matrix $K$ must be scaled. If you crop, $c_x, c_y$ shift. If you resize, $f_x, f_y$ scale).

### Practical Challenges

1.  **Write a Script:** Create `calibrate.py` that reads a folder of images and outputs `calibration_matrix.yaml`.
2.  **Visualize Distortion:** Create a "Mesh" grid and apply the distortion coefficients to see how it warps.

---

## 📚 Further Reading & Resources

### Documentation
*   **OpenCV Camera Calibration Tutorial.**
*   **Zhang's Method (The math behind it).**

---

## 🎓 Summary

Today we covered:
- ✅ **Pinhole Model:** The math of vision.
- ✅ **Intrinsics:** $f, c$.
- ✅ **Extrinsics:** $R, t$.
- ✅ **Distortion:** Barrel & Pincushion.
- ✅ **Target:** The importance of a flat board.

**Next:** Day 148 - Implementing OpenCV Calibration.

---

**Day 147 Complete** | Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision


