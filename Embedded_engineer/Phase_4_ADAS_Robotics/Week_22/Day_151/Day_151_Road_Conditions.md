# Day 151: Road Conditions (Construction, Potholes)
## Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases

---

> **📝 Day 151 Focus:**
> The map says "Go Straight", but there is a cone and a "Road Closed" sign. **Construction Zones** break all the rules. Lane lines are gone, signs are temporary, and the road is rough. Today, we learn to navigate **Unstructured Environments**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Detect** Construction Cones and Barrels using Color/Shape.
2.  **Identify** Drivable Area without lane markings (Free Space Detection).
3.  **Detect** Potholes using Stereo Vision / Lidar.
4.  **Handle** Detours (Map vs Perception conflict).
5.  **Implement** a Virtual Lane generator based on boundaries.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Computer Vision:** Color Thresholding (HSV).
-   **Day 123:** Semantic Segmentation.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `opencv-python`, `scikit-learn` (RANSAC).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Construction Zone

-   **Features:** Orange cones, Yellow barrels, Temporary signs, Workers.
-   **Challenge:** HD Map is useless (outdated). Lane lines are painted over or criss-crossed.
-   **Strategy:** Switch from **Map-Based** to **Perception-Based** navigation. Follow the "Corridor of Cones".

### 🔹 Part 2: Free Space Detection

If there are no lanes, where can I drive?
-   **Semantic Segmentation:** Classify pixels as "Road" vs "Not Road".
-   **Occupancy Grid:** Fuse Lidar/Camera to find empty space.
-   **Virtual Lane:** Fit a polynomial to the boundary of the free space (e.g., left cones and right curb).

### 🔹 Part 3: Pothole Detection

-   **Visual:** Dark blob, elliptical. Hard to distinguish from patches.
-   **3D (Lidar/Stereo):**
    -   Fit a plane to the road surface ($ax+by+cz+d=0$).
    -   Points significantly *below* the plane ($> 5cm$) are potholes.
    -   Points *above* are obstacles.

---

## 💻 Implementation: Cone Corridor Navigation

**Scenario:**
-   Car is in a construction zone.
-   Orange cones on Left and Right.
-   Task: Find the center path.

### 🛠️ Setup
Create `week22_day151` and `cone_nav.py`.

```bash
mkdir -p ~/ros2_ws/src/week22_day151
cd ~/ros2_ws/src/week22_day151
touch cone_nav.py
```

### 👨‍💻 Code: Virtual Lane from Cones

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

def detect_cones(img):
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Orange Color Range (Cones)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50: # Filter noise
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append([cx, cy])
                
    return np.array(centers)

def fit_corridor(centers, img_shape):
    if len(centers) < 2:
        return None, None
        
    # Split into Left and Right based on image center
    mid_x = img_shape[1] // 2
    left_cones = centers[centers[:, 0] < mid_x]
    right_cones = centers[centers[:, 0] > mid_x]
    
    # Fit Lines (y = mx + c) -> x = (y-c)/m
    # We fit x as function of y because lines are vertical-ish
    
    left_fit = None
    right_fit = None
    
    if len(left_cones) > 1:
        ransac = RANSACRegressor()
        ransac.fit(left_cones[:, 1].reshape(-1, 1), left_cones[:, 0])
        left_fit = ransac
        
    if len(right_cones) > 1:
        ransac = RANSACRegressor()
        ransac.fit(right_cones[:, 1].reshape(-1, 1), right_cones[:, 0])
        right_fit = ransac
        
    return left_fit, right_fit

def main():
    # 1. Create Synthetic Image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (100, 100, 100) # Road
    
    # Draw Cones (Orange Circles)
    # Left Boundary
    for y in range(50, 400, 50):
        x = 200 + np.random.randint(-10, 10)
        cv2.circle(img, (x, y), 10, (0, 165, 255), -1)
        
    # Right Boundary
    for y in range(50, 400, 50):
        x = 400 + np.random.randint(-10, 10)
        cv2.circle(img, (x, y), 10, (0, 165, 255), -1)
        
    # 2. Detect
    centers = detect_cones(img)
    
    # 3. Fit
    left_model, right_model = fit_corridor(centers, img.shape)
    
    # 4. Visualize
    y_plot = np.linspace(0, 400, 100).reshape(-1, 1)
    
    if left_model:
        x_left = left_model.predict(y_plot)
        pts_left = np.column_stack((x_left, y_plot)).astype(np.int32)
        cv2.polylines(img, [pts_left], False, (0, 255, 0), 2)
        
    if right_model:
        x_right = right_model.predict(y_plot)
        pts_right = np.column_stack((x_right, y_plot)).astype(np.int32)
        cv2.polylines(img, [pts_right], False, (0, 255, 0), 2)
        
    # Center Path
    if left_model and right_model:
        x_center = (x_left + x_right) / 2
        pts_center = np.column_stack((x_center, y_plot)).astype(np.int32)
        cv2.polylines(img, [pts_center], False, (0, 0, 255), 2)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Construction Zone Navigation")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Missing Cone

### Lab Objectives
1.  Run the script.
2.  **Observation:** The Green lines fit the cones, and the Red line shows the safe path.
3.  **Experiment:**
    -   Remove a few cones (simulate gaps).
    -   **Result:** RANSAC is robust and still finds the line.
    -   Add a random orange blob in the middle (worker's vest).
    -   **Result:** Simple logic might fail.
    -   **Fix:** Use temporal tracking. Cones don't move. Workers do.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Color Ambiguity
**Symptom:** Detecting tail lights as cones.
**Cause:** Red/Orange overlap in HSV.
**Solution:** Check Shape (Aspect Ratio). Cones are tall triangles. Tail lights are wide rectangles.

#### 2. Sharp Turns
**Symptom:** Linear fit fails on curves.
**Cause:** `y = mx + c` is a line.
**Solution:** Use `np.polyfit` (Degree 2 or 3) or Splines for curved roads.

---

## ⚡ Optimization & Best Practices

### 1. Voronoi Diagrams
For unstructured environments (parking lots, open plazas).
-   Compute Voronoi regions of obstacles.
-   The edges of the Voronoi diagram maximize clearance from all obstacles.
-   Safe path = Voronoi edges.

### 2. Negative Obstacle Detection
For potholes.
-   **Stereo Disparity:** Look for regions where disparity jumps "further away" than the ground plane.
-   **Flash Lidar:** Intensity return is low (water/dark) or geometry is concave.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is HD Map unreliable in construction zones?
    *   **A:** Construction changes daily. The map is static.
2.  **Q:** What is RANSAC?
    *   **A:** Random Sample Consensus. An iterative algorithm to fit a model (line) to data containing outliers (noise).
3.  **Q:** How do we handle "Road Closed" signs?
    *   **A:** OCR (Optical Character Recognition) + Object Detection. If detected, trigger a "Re-route" behavior.

### Challenge Task
**Task:** Pothole Simulator.
1.  Create a synthetic depth map (Plane).
2.  Subtract a Gaussian blob from the center (Pothole).
3.  Write a script to detect pixels where $Z < Z_{plane} - 0.05$.

---

## 📚 Further Reading & References
-   [RANSAC Algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus)
-   [Drivable Area Detection](https://arxiv.org/abs/1805.07029)

---

**Day 151 Complete** | Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases
