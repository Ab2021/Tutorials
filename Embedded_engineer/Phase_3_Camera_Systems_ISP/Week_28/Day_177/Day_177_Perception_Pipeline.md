# Day 177: Perception Pipeline (AI + CV)
## Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project

---

## 🎯 Learning Objectives
1.  **Implement** a Hybrid Pipeline: Classical CV (Lane Detection) + Deep Learning (Object Detection).
2.  **Optimize** for Concurrency: Run AI and CV in parallel threads.
3.  **Fuse** Data: Map 2D Bounding Boxes to 3D World Coordinates (using Camera Calibration).
4.  **Detect** QR Codes for "Mission Commands" (e.g., "Stop", "Turn Left").
5.  **Visualize** the "World State" overlay.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Robot with Camera.
*   **Software:** OpenCV, TensorRT (YOLO), PyZBar.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Hybrid Approach
*   **Why Hybrid?**
    *   **Lanes:** Classical CV (Canny/Hough) is faster and more precise for finding continuous lines than a generic object detector.
    *   **Objects:** AI (YOLO) is robust for detecting People/Boxes which vary in shape/texture.
*   **Synchronization:** CV runs at 30fps. AI runs at 15fps. We need to handle the rate mismatch (e.g., use the latest available AI result).

### 🔹 Part 2: Inverse Perspective Mapping (IPM)
*   **Goal:** Convert "Image Pixels" to "Floor Meters".
*   **Homography:** A 3x3 matrix that maps the ground plane in the image to a top-down view.
*   **Calibration:** Place a square on the floor. Click 4 corners in image. Map to $(0,0), (1,0), (1,1), (0,1)$ meters.

### 🔹 Part 3: QR Code Navigation
*   **Fiducial Markers:** QR Codes or ArUco markers act as "Traffic Signs".
*   **Data:** The QR code contains JSON: `{"action": "turn_left", "speed": 0.5}`.

---

## 💻 Implementation Examples

### Example 1: Lane Detection (Classical CV)

```python
import cv2
import numpy as np

def detect_lane(frame):
    # 1. ROI (Bottom half)
    h, w = frame.shape[:2]
    roi = frame[h//2:, :]
    
    # 2. Color Threshold (White/Yellow)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 3. Centroid
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Deviation from center (-1.0 to 1.0)
        deviation = (cx - w/2) / (w/2)
        return deviation
    return 0.0
```

### Example 2: The Perception Class (Threading)

```python
import threading
import time
from queue import Queue

class Perception:
    def __init__(self, camera, yolo_engine):
        self.camera = camera
        self.yolo = yolo_engine
        self.running = True
        self.latest_state = {"deviation": 0, "objects": []}
        
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        while self.running:
            frame = self.camera.get_frame()
            if frame is None: continue
            
            # 1. Lane (Fast)
            dev = detect_lane(frame)
            
            # 2. Objects (Slow - Run every 3rd frame?)
            objs = self.yolo.detect(frame)
            
            # 3. QR Codes
            qrs = detect_qr(frame)
            
            # Update State
            self.latest_state = {
                "deviation": dev,
                "objects": objs,
                "qrs": qrs
            }

    def get_state(self):
        return self.latest_state
```

### Example 3: Distance Estimation (Pinhole)

```python
KNOWN_WIDTH = 0.5 # meters (Width of box)
FOCAL_LENGTH = 800 # pixels

def get_distance(bbox_width_pixels):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width_pixels
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Calibrate Homography

**Objective:** Measure distance on floor.

**Steps:**
1.  Place tape on floor at 0.5m, 1.0m, 1.5m.
2.  Capture image.
3.  Find the pixel Y-coordinate of each tape line.
4.  Fit a curve (or Homography) to map $Y_{pixel} \to Z_{meters}$.

### Lab 2: Lane Keeping Test

**Objective:** Follow the line.

**Steps:**
1.  Put robot on track.
2.  Run `detect_lane`.
3.  Print `deviation`.
4.  **Verify:** Left of line -> Negative deviation. Right -> Positive.

### Lab 3: Stop Sign Detection

**Objective:** Safety.

**Steps:**
1.  Train YOLO on "Stop Sign".
2.  Place Stop Sign in front of robot.
3.  **Verify:** Robot detects it. Distance decreases as it approaches.

---

## 🐛 Debugging Perception

### Debug 1: "Lane Jitter"

**Symptom:** Deviation jumps wildly.

**Cause:**
*   Lighting changes (Shadows).
*   **Fix:** Use Edge Detection (Canny) instead of Color. Or use a Moving Average Filter on the deviation value.

### Debug 2: "Laggy Video"

**Symptom:** Robot reacts 1 second late.

**Cause:**
*   Queue buildup.
*   **Fix:** Ensure Camera HAL uses `appsink drop=1`. Ensure Perception thread doesn't sleep.

---

## ⚡ Performance Optimization

### Optimization 1: ROI (Region of Interest)

*   Don't run YOLO on the whole image if you only care about obstacles on the floor.
*   Crop the bottom 50%? Or maybe the center?
*   **Trade-off:** You might miss hanging obstacles.

### Optimization 2: MobileNet-SSD vs YOLO

*   If YOLOv8 is too slow (10fps), try MobileNet-SSD (30fps).
*   Less accurate, but speed is critical for control loops.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Homography"?** (A transformation that maps one plane to another. Used to remove perspective distortion).
2.  **Why do we need "Threads" here?** (If we run CV and AI sequentially in the main loop, the control loop will be blocked. Motors will stutter).
3.  **What is "Fiducial"?** (A reference marker placed in the scene to help the robot locate itself).

### Practical Challenges

1.  **Traffic Light:** Detect Red/Green color blobs. Stop on Red. Go on Green.
2.  **Tunnel:** Switch to "Night Mode" (High Gain) automatically when entering a dark tunnel.

---

## 📚 Further Reading & Resources

### Documentation
*   **OpenCV Image Processing.**
*   **PyZBar Documentation.**

---

## 🎓 Summary

Today we covered:
- ✅ **Hybrid Pipeline:** Combining CV and AI.
- ✅ **Lane Detection:** Finding the path.
- ✅ **IPM:** Pixels to Meters.
- ✅ **QR Codes:** Reading signs.
- ✅ **Threading:** Keeping it fast.

**Next:** Day 178 - Control & Logic (State Machine).

---

**Day 177 Complete** | Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project


