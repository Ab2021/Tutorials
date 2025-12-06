# Day 150: Lane Detection (PolyFit & Curves)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 22: Autonomous Driving Stack

---

> **📝 Content Creator Instructions:**
> The road is not a straight line.
> - **Focus:** Inverse Perspective Mapping (IPM), Sliding Window search, 3rd Order Polynomial Fitting ($y = ax^3 + bx^2 + cx + d$), and Calculating Radius of Curvature.
> - **Code:** A Python script `lane_fitter.py` that processes a binary lane map (BEV), fits polynomials to Left and Right lanes, and computes the Offset from Center.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Transform** a Camera images to Bird's Eye View (IPM) to make parallel lanes look parallel.
2.  **Algorithmize** the "Sliding Window" search to find lane pixels in a noisy image.
3.  **Regress** a 3rd Order Polynomial using Least Squares.
4.  **Compute** curvature $R = [1 + (y')^2]^{3/2} / |y''|$.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib opencv-python
```

### Prior Knowledge
- Camera Calibration (Day 15).
- Regression.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Perspective Problem

In a raw camera image, lanes vanish to a point.
*   **IPM (Inverse Perspective Mapping):** A Projective Transform (Homography) that "un-warps" the road plane.
*   **Result:** Lanes appear as parallel vertical lines. Good for fitting.

### 🔹 Part 2: The Model

Highways are designed using **Clothoids** (Euler Spirals), where curvature changes linearly.
Approximate Locally:
$$ x(y) = a y^3 + b y^2 + c y + d $$
*   Note: We fit $x$ as a function of $y$ because lanes are vertical.
*   **$d$:** Lateral Position (Offset).
*   **$c$:** Heading Angle.
*   **$b$:** Curvature.
*   **$a$:** Change in Curvature.

### 🔹 Part 3: Sliding Windows

How to find the pixels to fit?
1.  **Histogram:** Sum image columns. Peaks = Lane Centers.
2.  **Windows:** Place boxes at peaks at bottom.
3.  **Slide Up:** Move box up, recenter based on mean of pixels inside.
4.  **Collect:** Gather all pixels in boxes. Fit poly.

---

## 💻 Implementation: The Lane Tracker

We perform polyfit on a synthetic curved road.

### 🛠️ Project Structure
```text
day150_lanes/
├── src/
│   ├── lane_fitter.py
└── output/
    ├── fitted_lanes.png
```

### 👨‍💻 Polynomial Fitter (`src/lane_fitter.py`)

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_fake_lane_image():
    # Create an empty BEV image (Height x Width)
    h, w = 720, 1280
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Generate Synthetic Curved Lanes (x = ay^2 + by + c)
    # y coordinates (0 at top, 720 at bottom)
    plot_y = np.linspace(0, h-1, h)
    
    # Coefficients for a Left Turn
    radius = 3000 # pixels (fake units)
    # x = y^2 / (2R) roughly
    
    # Left Lane
    left_x = (plot_y**2) / (2*radius) + 200
    # Right Lane (Parallel)
    right_x = left_x + 600 # 600 px lane width
    
    # Draw points on image with noise
    for i, y in enumerate(plot_y):
        lx = int(left_x[i] + np.random.normal(0, 10))
        rx = int(right_x[i] + np.random.normal(0, 10))
        
        if 0 <= lx < w: cv2.circle(img, (lx, int(y)), 2, 255, -1)
        if 0 <= rx < w: cv2.circle(img, (rx, int(y)), 2, 255, -1)
        
    return img

def sliding_window_search(binary_warped):
    # 1. Histogram
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # 2. Config
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0]/nwindows)
    margin = 100
    minpix = 50
    
    # Indices
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = []
    right_lane_inds = []
    
    # Current positions
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Windows
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        
        # Identify nonzeros in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:            
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract fit points
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(leftx, lefty, rightx, righty, shape):
    # Fit 2nd order (for robustness) or 3rd
    # x = Ay^2 + By + C
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x values for plotting
    ploty = np.linspace(0, shape[0]-1, shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, right_fit, ploty, left_fitx, right_fitx

def main():
    img = generate_fake_lane_image()
    
    lx, ly, rx, ry, viz_img = sliding_window_search(img)
    
    if len(lx) == 0 or len(rx) == 0:
        print("Lanes not found.")
        return
        
    l_fit, r_fit, ploty, l_fitx, r_fitx = fit_polynomial(lx, ly, rx, ry, img.shape)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(viz_img)
    plt.plot(l_fitx, ploty, color='yellow', linewidth=3, label='Left Poly')
    plt.plot(r_fitx, ploty, color='yellow', linewidth=3, label='Right Poly')
    plt.title("Lane Detection (Sliding Window + Polyfit)")
    plt.legend()
    plt.savefig("output/fitted_lanes.png")
    
    # Curvature Calculation (at bottom of image)
    y_eval = np.max(ploty)
    # R = ((1 + (2Ay + B)^2)^1.5) / |2A|
    left_curverad = ((1 + (2*l_fit[0]*y_eval + l_fit[1])**2)**1.5) / np.absolute(2*l_fit[0])
    print(f"Radius of Curvature: {left_curverad:.1f} pixels")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Bad Marker"

### 1. Lab Objectives
- **Run:** Sim.
- **Problem:** Road markings are dashed (dotted lines).
- **Modify:** In `generate_fake_lane_image`, make the Left Lane dashed (only draw every 50 pixels).
- **Run:** Does Sliding Window still work?
- **Result:** Yes, because the window is tall enough to catch the next dash.
- **Fail:** Make gaps 200px. Window loses the lane.
- **Fix:** "Search from Prior". If we found the lane in Frame $t-1$, strictly search around that poly in Frame $t$. Don't use Histogram again.

---

## 🚀 Project: "Lane Keeping Assist (LKA)"

**Goal:** Calculate Steering Angle.
1.  **Center:** $x_{center} = (x_{left} + x_{right}) / 2$ (at bottom).
2.  **Error:** $e = x_{center} - (ImageWidth / 2)$.
3.  **Control:** $\delta = -K_p \cdot e$.
4.  **Result:** Use `ego_vehicle` from Day 148 to close the loop.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Wobbly Lines"
*   **Cause:** Fitting high-order poly to noisy pixels.
*   **Fix:** Use Weighted Least Squares. Or Average coefficients over 5 frames (Smoothing).

#### 2. "Perspective Fail"
*   **Cause:** Pitching car (Braking). IPM assumption (flat ground) breaks. Lines converge.
*   **Fix:** Adaptive IPM (Horizon estimation). Or use 3D Lane Net.

---

## ⚡ Optimization: Spatial Consistency

Lanes are parallel.
*   **Constraint:** $a_{left} \approx a_{right}$ and $b_{left} \approx b_{right}$.
*   **Fit:** Fit *both* lanes simultaneously with shared curvature parameters.
    $$ Loss = \sum (x_{L} - P_L(y))^2 + \sum (x_{R} - P_R(y))^2 + \lambda (a_L - a_R)^2 $$

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why 3rd Order?
    *   **A:** 2nd Order is a Parabola (Constant curvature change). 3rd order allows curvature to *change rate* (Clothoid transition).
2.  **Q:** What is the unit of Curvature?
    *   **A:** $1/meter$. Radius is $meter$. Straight road has Radius = Infinity, Curvature = 0.
3.  **Q:** Deep Learning vs PolyFit?
    *   **A:** PolyFit is robust on highways with good markings. DL (LaneNet) is needed for city streets with no markings (follow the curb).

### Challenge Task
> **Task:** Merge/Split.
> 1. Simulate a highway exit (Y-split).
> 2. You now have 3 lines (Left, Right, Exit).
> 3. Modify Sliding Window to track *multiple* peaks.

---

## 📚 Further Reading
- **Udacity Self-Driving Car ND:** Advanced Lane Finding module.
- **Papers:** "Bézier Curve Lane Detection".

---

**Day 150 Complete**
