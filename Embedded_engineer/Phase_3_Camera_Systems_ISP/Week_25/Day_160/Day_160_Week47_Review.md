# Day 160: Week 25 Review & Project (The Smart Security Camera)
## Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems

---

## 🎯 Learning Objectives
1.  **Integrate** the ML components (Detection, Tracking, Logic) into a complete application.
2.  **Build** a "Smart Security Camera" that detects people, tracks them, and alerts if they enter a "Restricted Zone".
3.  **Optimize** the pipeline for 30 FPS on embedded hardware.
4.  **Validate** the system in real-world scenarios (Day/Night, Occlusion).
5.  **Prepare** for Week 48 (Performance Optimization).

---

## 📚 Week 25 Recap

### Topics Covered

**Day 154: Edge AI Basics**
- TensorRT vs TFLite, ONNX conversion.

**Day 155: Object Detection**
- YOLOv8, NMS, Bounding Boxes.

**Day 156: Semantic Segmentation**
- UNet, Pixel-level classification, Privacy masking.

**Day 157: Pose Estimation**
- MoveNet, Skeleton tracking, Action recognition.

**Day 158: Dataset Generation**
- Collection, Annotation (YOLO format), Augmentation.

**Day 159: Model Optimization**
- Quantization (INT8), Pruning, Latency reduction.

---

## 💻 Week 25 Project: The Smart Security Camera

### Objective
Create a Python application that runs on a Jetson/Pi/Laptop. It reads the camera stream, detects people, tracks them, and checks if they cross a virtual line.

### Features
1.  **Person Detection:** Use YOLOv8n (Quantized).
2.  **Tracking:** Use ByteTrack (or simple IoU tracker) to assign IDs to people.
3.  **Zone Monitoring:** Define a polygon (e.g., "The Driveway"). If a person's foot (bottom center of box) is inside, turn the box RED.
4.  **Privacy Mode:** Blur faces (using the bounding box).
5.  **Alerting:** Print "ALARM: Person ID 42 in Zone!" to console (or MQTT).

### Architecture
*   **Thread 1 (Capture):** Reads frames from Camera -> Queue.
*   **Thread 2 (Inference):** Reads Queue -> Preprocess -> TensorRT Engine -> Postprocess (NMS) -> Queue.
*   **Thread 3 (Logic & Display):** Reads Detections -> Update Tracker -> Check Zones -> Draw UI -> Show.

### Implementation Snippet (Zone Check)

```python
import cv2
import numpy as np
from shapely.geometry import Point, Polygon

# Define Zone (Polygon)
zone_pts = np.array([[100, 100], [500, 100], [500, 400], [100, 400]])
zone_poly = Polygon(zone_pts)

def check_zone(detections):
    """
    detections: List of [x1, y1, x2, y2, conf, class_id]
    """
    alerts = []
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        
        # Point of interest: Bottom Center (The feet)
        foot_x = (x1 + x2) / 2
        foot_y = y2
        point = Point(foot_x, foot_y)
        
        if zone_poly.contains(point):
            alerts.append(det)
            
    return alerts

def draw_zone(frame, alerts):
    # Draw Zone
    cv2.polylines(frame, [zone_pts], True, (255, 0, 0), 2)
    
    # Draw Alerts
    for det in alerts:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "INTRUDER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    return frame
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Define the Zone

**Objective:** Interactive Setup.

**Steps:**
1.  Write a script that captures one frame.
2.  Use `cv2.setMouseCallback` to let the user click 4 points.
3.  Save these points to `zone_config.json`.
4.  Load this config in the main app.

### Lab 2: Face Blurring (Privacy)

**Objective:** GDPR Compliance.

**Steps:**
1.  Inside the Person Box, run a secondary Face Detector? Too slow.
2.  **Heuristic:** Assume the face is in the top 20% of the Person Box.
3.  **Blur:** `roi = frame[y1:y1+h//5, x1:x2]`; `frame[...] = cv2.GaussianBlur(roi, ...)`

### Lab 3: Stress Test

**Objective:** Max FPS.

**Steps:**
1.  Run the app.
2.  Measure FPS.
3.  **Profile:** Is the bottleneck Capture, Inference, or Drawing?
4.  **Optimize:** If Drawing is slow, reduce the resolution of the UI overlay. If Inference is slow, use a smaller model.

---

## 📝 Assessment Questions

### Comprehensive Questions

1.  **Why do we track "Feet" instead of "Center"?** (Because the camera is usually high up looking down. The feet touch the ground plane where the zone is defined. The head might appear "outside" the zone due to perspective).
2.  **What is "ByteTrack"?** (A tracking algorithm that associates high-confidence detections first, then tries to match low-confidence detections to existing tracks. Robust to occlusion).
3.  **How to handle "Ghost" detections?** (Require a track to be active for N frames before triggering an alarm).

### Practical Challenges

1.  **Loitering Detection:** Trigger alarm only if a person stays in the zone for > 5 seconds.
2.  **Line Crossing:** Count people entering vs leaving a room. (Vector cross product of movement vector and line vector).

---

## 📚 Resources & Next Steps

### Week 25 Summary

**Completed:**
- ✅ **Edge AI:** Running models on small chips.
- ✅ **Detection/Seg/Pose:** The "Big Three" tasks.
- ✅ **Data:** Creating custom datasets.
- ✅ **Optimization:** Making it fast.
- ✅ **Project:** A real-world security app.

### Week 48 Preview (Performance & Power)

**Topics:**
- **Profiling:** Finding bottlenecks.
- **Zero-Copy:** Efficient memory.
- **Power:** Measuring Watts.
- **Thermal:** Keeping it cool.
- **Boot Time:** Starting fast.

---

**Day 160 Complete** | Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems


