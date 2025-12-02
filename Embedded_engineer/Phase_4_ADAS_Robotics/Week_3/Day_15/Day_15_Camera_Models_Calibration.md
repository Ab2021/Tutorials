# Day 15: Camera Models & Calibration
## Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning

---

> **📝 Day 15 Focus:**
> Cameras are the primary sensor for human-like perception in ADAS. However, a raw image is just a 2D array of pixels. To map these pixels to the 3D world (and vice versa), we must understand the **Camera Model** and remove optical distortions through **Calibration**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Derive** the Pinhole Camera Model and the Intrinsic Matrix ($K$).
2.  **Differentiate** between Intrinsic (Internal) and Extrinsic (Pose) parameters.
3.  **Model** Radial and Tangential lens distortion coefficients.
4.  **Implement** a robust Camera Calibration pipeline using OpenCV and a Checkerboard.
5.  **Undistort** live video streams and project 3D points onto the 2D image plane.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Linear Algebra:** Matrix multiplication, Homogeneous Coordinates.
-   **Optics:** Basic understanding of lenses (focal length).
-   **Python:** OpenCV (`cv2`), NumPy.

### Hardware Requirements
-   **Camera:** USB Webcam or Raspberry Pi Camera.
-   **Target:** A printed Checkerboard pattern (e.g., 9x6).

### Software Stack
-   **Python Libraries:** `opencv-python`, `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Pinhole Camera Model

The Pinhole model describes how a 3D point $(X, Y, Z)$ is projected onto a 2D image plane $(u, v)$.

#### 1.1 The Projection Equation
$$ s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} R & t \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} $$

-   **$s$:** Scale factor (Depth $Z$).
-   **$\mathbf{K}$:** Intrinsic Matrix (Internal properties).
-   **$[R|t]$:** Extrinsic Matrix (Pose of camera in world).

#### 1.2 The Intrinsic Matrix ($\mathbf{K}$)
$$ \mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} $$

-   **$f_x, f_y$:** Focal length in pixels.
    -   $f_{pixel} = \frac{f_{mm}}{pixel\_size_{mm}}$.
-   **$c_x, c_y$:** Principal Point (Optical center, usually image center).
-   **Skew:** Usually 0 (pixels are square).

#### 1.3 The Extrinsic Matrix ($[R|t]$)
Describes the transformation from World Coordinates to Camera Coordinates.
-   **$R$ (3x3):** Rotation matrix.
-   **$t$ (3x1):** Translation vector.

---

### 🔹 Part 2: Lens Distortion

Real lenses are not perfect pinholes. They bend light, causing straight lines to appear curved.

#### 2.1 Radial Distortion
Caused by the shape of the lens.
-   **Barrel Distortion:** Image bulges out ($k < 0$).
-   **Pincushion Distortion:** Image pinches in ($k > 0$).

Equation:
$$ x_{distorted} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) $$
$$ y_{distorted} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) $$

#### 2.2 Tangential Distortion
Caused by the lens not being perfectly parallel to the image sensor.
Equation:
$$ x_{distorted} = x + [2p_1 xy + p_2(r^2 + 2x^2)] $$
$$ y_{distorted} = y + [p_1(r^2 + 2y^2) + 2p_2 xy] $$

**Distortion Coefficients:** $D = [k_1, k_2, p_1, p_2, k_3]$.

---

### 🔹 Part 3: Calibration Process

How do we find $K$ and $D$?
We show the camera a known object (Checkerboard).
1.  **Detect Corners:** Find the 2D pixel coordinates $(u, v)$ of the checkerboard corners.
2.  **Known 3D Points:** We define the 3D world coordinates $(X, Y, 0)$ of the corners (assuming $Z=0$ on the board).
3.  **Optimize:** Solve for $K, D, R, t$ that minimizes the reprojection error (difference between detected corners and projected 3D points).

---

## 💻 Implementation: Camera Calibration Tool

We will build a Python tool `calibrate.py` that:
1.  Captures images from a webcam.
2.  Detects checkerboard corners.
3.  Calibrates the camera.
4.  Saves the parameters to a YAML file.
5.  Undistorts the live feed.

### 🛠️ Setup
Create `week3_day15` and `calibrate.py`.

```bash
mkdir -p ~/ros2_ws/src/week3_day15
cd ~/ros2_ws/src/week3_day15
touch calibrate.py
```

### 👨‍💻 Code: Calibration Script

```python
import numpy as np
import cv2
import glob
import yaml
import os

class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        self.CHECKERBOARD = checkerboard_size # (cols, rows) internal corners
        self.SQUARE_SIZE = square_size # meters
        
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= self.SQUARE_SIZE

    def capture_images(self):
        print("Press 'c' to capture, 'q' to quit.")
        cap = cv2.VideoCapture(0)
        
        if not os.path.exists('calib_images'):
            os.makedirs('calib_images')
            
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, None)
            
            display = frame.copy()
            
            if ret:
                cv2.drawChessboardCorners(display, self.CHECKERBOARD, corners, ret)
                cv2.putText(display, "Pattern Detected! Press 'c'", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Calibration Capture', display)
            
            key = cv2.waitKey(1)
            if key == ord('c') and ret:
                fname = f"calib_images/img_{count:03d}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
                count += 1
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def calibrate(self):
        images = glob.glob('calib_images/*.jpg')
        if not images:
            print("No images found in calib_images/")
            return
            
        print(f"Found {len(images)} images. Processing...")
        
        img_shape = None
        
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape[::-1]
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, None)
            
            if ret:
                # Refine corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)
                
        print("Calibrating camera... (this may take a moment)")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_shape, None, None)
            
        print(f"RMS Error: {ret:.4f}")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coeffs:\n", dist)
        
        # Save to YAML
        data = {
            'camera_matrix': mtx.tolist(),
            'dist_coeff': dist.tolist(),
            'rms_error': ret
        }
        with open('calibration.yaml', 'w') as f:
            yaml.dump(data, f)
        print("Saved to calibration.yaml")
        
        return mtx, dist

    def undistort_live(self, mtx, dist):
        cap = cv2.VideoCapture(0)
        
        # Optimize camera matrix
        h, w = int(cap.get(4)), int(cap.get(3))
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Undistort
            dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            
            # Crop
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            
            cv2.imshow('Original', frame)
            cv2.imshow('Undistorted', dst)
            
            if cv2.waitKey(1) == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calib = CameraCalibrator()
    
    # 1. Capture
    # calib.capture_images() 
    
    # 2. Calibrate
    # mtx, dist = calib.calibrate()
    
    # 3. Test (Load if exists)
    if os.path.exists('calibration.yaml'):
        with open('calibration.yaml', 'r') as f:
            data = yaml.safe_load(f)
            mtx = np.array(data['camera_matrix'])
            dist = np.array(data['dist_coeff'])
            calib.undistort_live(mtx, dist)
    else:
        print("Please capture and calibrate first.")
```

---

## 🔬 Lab Exercise: Reprojection Error

### Lab Objectives
1.  Print a checkerboard and stick it to a flat surface.
2.  Run `capture_images()` and take 20-30 photos from different angles/distances.
    -   *Tip:* Ensure corners are visible. Don't move too fast (blur).
3.  Run `calibrate()`.
4.  **Analyze:** Look at the RMS Error.
    -   < 0.5 pixels: Excellent.
    -   0.5 - 1.0 pixels: Good.
    -   > 1.0 pixels: Bad. Retake images.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Checkerboard Not Detected
**Symptom:** `findChessboardCorners` returns False.
**Cause:**
-   Lighting is too bright (glare) or too dark.
-   Pattern is partially occluded.
-   Wrong dimensions (Count *internal* corners, not squares!).
    -   If board is 10x7 squares, corners are 9x6.

#### 2. High RMS Error
**Cause:**
-   Poor quality images (blur).
-   Checkerboard not flat (paper bending).
-   Not enough variety in angles (all images look the same).

#### 3. Undistorted Image looks weird (Black borders)
**Cause:** `getOptimalNewCameraMatrix` alpha parameter.
-   `alpha=0`: Crop all invalid pixels (Zoom in).
-   `alpha=1`: Keep all pixels (Black borders).

---

## ⚡ Optimization & Best Practices

### 1. Charuco Boards
Standard checkerboards suffer from occlusion (if one corner is hidden, detection fails).
**Charuco Boards** (Checkerboard + ArUco markers) allow detection even if part of the board is occluded. They are the industry standard for robust calibration.

### 2. Stereo Calibration
For stereo cameras, you need to calibrate both cameras *and* the relationship ($R, t$) between them.
-   `cv2.stereoCalibrate` uses the same image pairs to find the extrinsic transform.

### 3. Fisheye Model
For wide-angle lenses (>120 deg), the standard distortion model fails. Use `cv2.fisheye` module which implements the Equidistant projection model.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the Principal Point ($c_x, c_y$)?
    *   **A:** The point where the optical axis intersects the image plane. Usually the center of the image.
2.  **Q:** Why do we need to calibrate a camera?
    *   **A:** To correct lens distortion and to determine the focal length for 3D measurement.
3.  **Q:** What happens if you use a calibrated camera matrix on a resized image?
    *   **A:** You must scale $f_x, f_y, c_x, c_y$ by the same resize factor!

### Challenge Task
**Task:** Project a 3D Cube.
1.  Use `cv2.solvePnP` to find the pose ($R, t$) of the checkerboard in the current frame.
2.  Define 3D points for a cube standing on the board.
3.  Project them to 2D using `cv2.projectPoints`.
4.  Draw lines to visualize the cube (Augmented Reality).

---

## 📚 Further Reading & References
-   [OpenCV Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
-   [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

---

**Day 15 Complete** | Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning
