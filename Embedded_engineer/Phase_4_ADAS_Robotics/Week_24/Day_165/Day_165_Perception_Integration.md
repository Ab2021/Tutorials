# Day 165: Perception Integration
## Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking

---

> **📝 Day 165 Focus:**
> You can't park by feel. You need to see the lines. **AVP Perception** is different from Highway Perception. We care about **360° Close-Range** visibility. We fuse Fisheye Cameras (Surround View) and Ultrasonics to find the slot and avoid the pillar.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Implement** Inverse Perspective Mapping (IPM) for Bird's Eye View.
2.  **Detect** Parking Lines using semantic segmentation or edge filters.
3.  **Fuse** Ultrasonic data into an Occupancy Grid.
4.  **Detect** Empty Slots using bounding box logic.
5.  **Visualize** the fused "World Model" in Rviz.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 125:** Sensor Fusion (BEV).
-   **Day 163:** Environment Mapping.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **OpenCV:** Image processing.
-   **ROS 2:** `sensor_msgs/Range`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Surround View Monitoring (SVM)

-   **4 Cameras:** Front, Rear, Left Mirror, Right Mirror.
-   **Fisheye Distortion:** Must be corrected (undistorted) first.
-   **Stitching:** Images are projected onto a "Bowl" or ground plane (IPM) and blended to create a single top-down view.

### 🔹 Part 2: Parking Slot Detection

Two approaches:
1.  **Vision-Based:** Detect white lines (corners).
    -   *Pros:* Precise alignment.
    -   *Cons:* Fails if lines are faded or covered by snow.
2.  **Free-Space Based:** Detect gap between two parked cars (Ultrasonics/Lidar).
    -   *Pros:* Robust to lighting.
    -   *Cons:* Depends on other cars parking correctly.

### 🔹 Part 3: Ultrasonic Fusion

-   **Sensor:** Returns a single scalar `range` (e.g., 1.5m).
-   **Field of View:** Wide cone (~60°).
-   **Mapping:** We don't know *where* in the cone the object is. We model it as an arc of probability in the Occupancy Grid.

---

## 💻 Implementation: Parking Line Detector

**Scenario:**
-   Input: Top-down (IPM) image of a parking spot.
-   Task: Find the 4 corners of the slot.

### 🛠️ Setup
Create `week24_capstone/perception`.

### 👨‍💻 Code: Line Detection (OpenCV)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_parking_lines(img):
    # 1. Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Edge Detection
    edges = cv2.Canny(blur, 50, 150)
    
    # 3. Hough Lines (Probabilistic)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    line_img = np.zeros_like(img)
    
    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter vertical/horizontal lines only (Parking structure)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            if abs(angle) < 10 or abs(abs(angle)-90) < 10:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detected_lines.append(line[0])
                
    return line_img, detected_lines

def find_corners(lines):
    # Find intersections of H and V lines
    # Simplified logic for demo
    corners = []
    # ... (Intersection math)
    return corners

def main():
    # 1. Create Synthetic Parking Spot Image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (50, 50, 50) # Asphalt
    
    # Draw White Lines (U-shape)
    cv2.line(img, (100, 100), (100, 300), (255, 255, 255), 5) # Left
    cv2.line(img, (300, 100), (300, 300), (255, 255, 255), 5) # Right
    cv2.line(img, (100, 300), (300, 300), (255, 255, 255), 5) # Back
    
    # Add Noise
    noise = np.random.randint(0, 50, (400, 400, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # 2. Detect
    result, lines = detect_parking_lines(img)
    
    # 3. Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input (IPM)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("Detected Lines")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Ultrasonic Grid Mapping

### Lab Objectives
1.  **Simulate USS:**
    -   Create a node that publishes `sensor_msgs/Range`.
    -   Range = 2.0m. FOV = 0.5 rad.
2.  **Update Grid:**
    -   For each USS reading, update the Occupancy Grid.
    -   **Inverse Sensor Model:**
        -   Cells *inside* the cone (d < range) $\to$ Free (Probability decreases).
        -   Cells *on* the arc (d = range) $\to$ Occupied (Probability increases).
3.  **Visualize:**
    -   View the grid in Rviz. You should see "Arc" shapes appearing where obstacles are detected.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Ghost Lines
**Symptom:** Detecting cracks in the pavement as parking lines.
**Cause:** Canny threshold too low.
**Solution:** Use **Semantic Segmentation** (UNet) trained on parking lots. It learns the *context* of a line, not just the gradient.

#### 2. Ultrasonic Cross-Talk
**Symptom:** Sensor A receives the echo from Sensor B.
**Cause:** Firing all sensors simultaneously.
**Solution:** Fire them sequentially (Round Robin) or use different frequencies (coding).

---

## ⚡ Optimization & Best Practices

### 1. Vacancy Detection (Deep Learning)
-   Train a CNN (e.g., YOLO) to detect "Empty Spot" directly.
-   Input: Fisheye image.
-   Output: Bounding Box + Orientation + Type (Parallel/Perpendicular).
-   Much more robust than line detection.

### 2. Multi-Sensor Fusion
-   Combine Vision (Lines) + Ultrasonics (Obstacles).
-   If Vision says "Spot Here" but USS says "Obstacle Here" (e.g., a shopping cart), mark as Occupied.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is IPM?
    *   **A:** Inverse Perspective Mapping. Removes the perspective effect to create a top-down view, making parallel lines actually parallel.
2.  **Q:** Why are Ultrasonics used for parking?
    *   **A:** They are cheap, robust to lighting (work in pitch black), and have very short minimum range (15cm), unlike Lidar/Radar.
3.  **Q:** How do we detect a "Cross-Junction" in a parking lot?
    *   **A:** Intersection of 4 lines. Or specific markings (Arrows).

### Challenge Task
**Task:** Spot Tracking.
1.  As the car moves, the detected spot moves in the image.
2.  Use a Kalman Filter to track the spot's position in the **Map Frame**.
3.  This stabilizes the target for the Planner.

---

## 📚 Further Reading & References
-   [Surround View System (TI)](https://www.ti.com/lit/an/spracg5/spracg5.pdf)
-   [Parking Slot Detection (Paper)](https://arxiv.org/abs/2006.00222)

---

**Day 165 Complete** | Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking
