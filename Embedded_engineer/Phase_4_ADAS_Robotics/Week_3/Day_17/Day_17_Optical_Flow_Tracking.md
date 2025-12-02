# Day 17: Optical Flow & Tracking
## Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning

---

> **📝 Day 17 Focus:**
> Feature matching (Day 16) is great for large movements, but for smooth video tracking, we use **Optical Flow**. It estimates the motion of pixels between consecutive frames based on the assumption that pixel intensity doesn't change much. This is the basis for Visual Odometry and Object Tracking.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Brightness Constancy Assumption and the Optical Flow equation.
2.  **Implement** Sparse Optical Flow using the Lucas-Kanade (LK) method.
3.  **Implement** Dense Optical Flow using the Farneback method.
4.  **Develop** a KLT (Kanade-Lucas-Tomasi) Tracker to track feature points across a video.
5.  **Analyze** the limitations of Optical Flow (Aperture problem, large displacements).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Calculus:** Taylor Series expansion (again!).
-   **Linear Algebra:** Least Squares.
-   **Day 16:** Feature Detection (Good Features to Track).

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).
-   **Camera:** Webcam for real-time testing.

### Software Stack
-   **Python Libraries:** `opencv-python`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Optical Flow Equation

#### 1.1 Brightness Constancy Assumption
We assume that the intensity of a pixel $I(x, y, t)$ stays constant as it moves to $(x+dx, y+dy)$ at time $t+dt$.

$$ I(x, y, t) = I(x+dx, y+dy, t+dt) $$

Using Taylor Series expansion on the right side:
$$ I(x+dx, y+dy, t+dt) \approx I(x,y,t) + \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt $$

Combining these:
$$ \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt = 0 $$

Dividing by $dt$:
$$ I_x u + I_y v + I_t = 0 $$

-   $I_x, I_y$: Spatial gradients (Sobel).
-   $I_t$: Temporal gradient (Frame difference).
-   $u, v$: Velocity vector ($\frac{dx}{dt}, \frac{dy}{dt}$). **This is what we want to find.**

#### 1.2 The Aperture Problem
We have one equation with two unknowns ($u, v$). We cannot solve it uniquely.
*   *Intuition:* Looking through a small hole (aperture), you can't tell if a line is moving along itself or staying still.

---

### 🔹 Part 2: Lucas-Kanade (Sparse Flow)

To solve the Aperture Problem, Lucas-Kanade assumes that the flow $(u, v)$ is **constant** in a small window (e.g., 3x3) around the pixel.

For a 3x3 window (9 pixels), we get 9 equations:
$$ \begin{bmatrix} I_{x1} & I_{y1} \\ \vdots & \vdots \\ I_{x9} & I_{y9} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} -I_{t1} \\ \vdots \\ -I_{t9} \end{bmatrix} $$

$$ A \mathbf{d} = \mathbf{b} $$

Solve using Least Squares:
$$ \mathbf{d} = (A^T A)^{-1} A^T \mathbf{b} $$

Note: $A^T A$ is exactly the Structure Tensor from Harris Corner Detector!
*   *Conclusion:* Optical Flow works best at **Corners**.

#### 2.1 Pyramidal LK
The standard LK method fails if motion is large (pixel moves outside the window).
**Solution:** Image Pyramids.
1.  Downscale image (Blur + Subsample).
2.  Compute flow at coarse level (small motion).
3.  Upscale flow and refine at finer level.

---

### 🔹 Part 3: Dense Optical Flow (Farneback)

Computes flow for **every** pixel.
-   Approximates image patches with quadratic polynomials.
-   Observes how the polynomial transforms under translation.
-   **Result:** A dense vector field. Useful for segmentation (moving object vs background).

---

## 💻 Implementation: Real-Time Tracking

We will implement two scripts:
1.  `sparse_flow.py`: Tracks specific points (KLT Tracker).
2.  `dense_flow.py`: Visualizes motion of the whole scene (HSV Color Map).

### 🛠️ Setup
Create `week3_day17` and the files.

```bash
mkdir -p ~/ros2_ws/src/week3_day17
cd ~/ros2_ws/src/week3_day17
touch sparse_flow.py dense_flow.py
```

### 👨‍💻 Code: Sparse Flow (KLT Tracker)

```python
import numpy as np
import cv2

def run_sparse_flow():
    cap = cv2.VideoCapture(0)

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    print("Press 'r' to reset features, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points (status == 1)
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                
                img = cv2.add(frame, mask)
                cv2.imshow('Sparse Optical Flow (KLT)', img)

                # Update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = None
        else:
            cv2.imshow('Sparse Optical Flow (KLT)', frame)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('r') or p0 is None or len(p0) < 5:
            # Re-detect features
            mask = np.zeros_like(old_frame)
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            print("Features reset.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_sparse_flow()
```

### 👨‍💻 Code: Dense Flow (Farneback)

```python
import numpy as np
import cv2

def run_dense_flow():
    cap = cv2.VideoCapture(0)
    
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # HSV Mask
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255 # Saturation

    print("Press 'q' to quit.")

    while True:
        ret, frame2 = cap.read()
        if not ret: break
        
        next_ = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate Dense Flow
        flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        # Convert flow to polar coordinates (magnitude, angle)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Map Angle to Hue
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # Map Magnitude to Value (Brightness)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        cv2.imshow('Dense Optical Flow', bgr)
        
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
            
        prvs = next_

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dense_flow()
```

---

## 🔬 Lab Exercise: Visual Odometry (Concept)

### Lab Objectives
1.  Run `sparse_flow.py`.
2.  Move the camera forward.
    -   *Observation:* Features move radially outward (Expansion).
3.  Rotate the camera left.
    -   *Observation:* Features move right.
4.  **Concept:** If we know the camera calibration ($K$), we can mathematically invert this flow field to calculate the camera's motion ($R, t$). This is **Visual Odometry**.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Tracking Lost
**Symptom:** Points disappear quickly.
**Cause:**
-   Fast motion (blur).
-   Lighting change (violates Brightness Constancy).
-   Feature moved out of frame.
**Solution:** Re-detect features periodically (as implemented in the code).

#### 2. Dense Flow is Noisy
**Symptom:** Speckles of color everywhere even when still.
**Cause:** Sensor noise.
**Solution:** Apply Gaussian Blur to frames before computing flow. Increase `winsize`.

#### 3. "Ghosting"
**Symptom:** Trails left behind.
**Cause:** The mask is not being cleared.
**Solution:** `mask = np.zeros_like(frame)` periodically.

---

## ⚡ Optimization & Best Practices

### 1. Inverse Compositional LK
A variant of LK that pre-computes the gradient of the *template* (old frame) instead of the image. Much faster iterations.

### 2. DIS Optical Flow
**Dense Inverse Search.** A modern, fast dense flow algorithm available in OpenCV (`cv2.DISOpticalFlow_create()`). It's faster than Farneback and often more accurate.

### 3. Hardware Optical Flow
NVIDIA Jetson and some sensors (PMW3901) have hardware accelerators for optical flow. Use them!

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the Brightness Constancy Assumption?
    *   **A:** Pixel intensity stays the same as it moves between frames.
2.  **Q:** Why does LK fail on flat walls?
    *   **A:** The Aperture Problem. $A^T A$ is not invertible (eigenvalues $\approx 0$).
3.  **Q:** What is the difference between Sparse and Dense flow?
    *   **A:** Sparse tracks specific points (fast). Dense calculates motion for every pixel (slow, detailed).

### Challenge Task
**Task:** Object Segmentation using Flow.
1.  Run Dense Flow.
2.  Threshold the Magnitude.
3.  If magnitude > threshold, mark pixel as "Moving Object".
4.  Draw a bounding box around the moving region.

---

## 📚 Further Reading & References
-   [Lucas-Kanade Paper (1981)](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_2/lucas_bruce_d_1981_2.pdf)
-   [OpenCV Optical Flow Tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)

---

**Day 17 Complete** | Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning
