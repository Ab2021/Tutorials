# Day 122: Object Detection (YOLO/SSD)
## Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception

---

> **📝 Day 122 Focus:**
> Classification says "This is a car". Detection says "There is a car at [x, y, w, h]". For ADAS, we need Detection. **YOLO (You Only Look Once)** is the industry standard for real-time detection.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the Object Detection task (Regression + Classification).
2.  **Calculate** IoU (Intersection over Union) to measure accuracy.
3.  **Explain** Anchor Boxes and Grid Cells.
4.  **Apply** Non-Maximum Suppression (NMS) to clean up duplicate boxes.
5.  **Deploy** a pre-trained YOLOv8 model on a video stream.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 121:** CNNs.
-   **Geometry:** Bounding Boxes.

### Hardware Requirements
-   **GPU:** Recommended for real-time inference.

### Software Stack
-   **Python:** `ultralytics` (YOLOv8), `opencv-python`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Grid Approach (YOLO)

YOLO divides the image into an $S \times S$ grid (e.g., $13 \times 13$).
-   If the center of an object falls into a grid cell, that cell is responsible for detecting it.
-   Each cell predicts $B$ bounding boxes and $C$ class probabilities.
-   **Output Tensor:** $S \times S \times (B \times 5 + C)$.
    -   5 comes from $(x, y, w, h, confidence)$.

### 🔹 Part 2: IoU (Intersection over Union)

How do we know if a prediction is correct?
$$ IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}} $$
-   $IoU > 0.5$: Usually considered a "True Positive".
-   $IoU = 1.0$: Perfect match.

### 🔹 Part 3: Non-Maximum Suppression (NMS)

YOLO often predicts multiple boxes for the same car.
**Algorithm:**
1.  Discard boxes with confidence < Threshold.
2.  Pick the box with highest confidence.
3.  Discard any remaining box with $IoU > 0.5$ with the picked box (Duplicate).
4.  Repeat.

---

## 💻 Implementation: YOLOv8 Car Detector

**Scenario:**
-   Input: Dashcam video (or static image).
-   Task: Detect Cars, Trucks, Pedestrians.
-   Output: Draw bounding boxes.

### 🛠️ Setup
Create `week18_day122` and `yolo_detect.py`.

```bash
mkdir -p ~/ros2_ws/src/week18_day122
cd ~/ros2_ws/src/week18_day122
pip install ultralytics opencv-python
touch yolo_detect.py
```

### 👨‍💻 Code: Real-Time Detection

```python
import cv2
from ultralytics import YOLO
import numpy as np

def main():
    # 1. Load Model
    # 'yolov8n.pt' is the Nano version (Fastest, least accurate)
    # 'yolov8x.pt' is the Xtra Large version (Slowest, most accurate)
    print("Loading YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt') 
    
    # 2. Load Image/Video
    # We create a dummy image for demo if no video file
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1) # Blue Box (Car?)
    cv2.rectangle(img, (400, 400), (450, 500), (0, 255, 0), -1) # Green Box (Pedestrian?)
    
    # In real lab, use: cap = cv2.VideoCapture('highway.mp4')
    
    print("Running Inference...")
    results = model(img)
    
    # 3. Process Results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Confidence
            conf = box.conf[0].cpu().numpy()
            
            # Class ID
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls]
            
            print(f"Detected {class_name} at [{x1:.0f}, {y1:.0f}] with Conf {conf:.2f}")
            
            # Draw
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    # Show
    cv2.imshow("YOLOv8 Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Confidence Threshold

### Lab Objectives
1.  Run the script.
2.  **Observation:** Since the image is just blue/green rectangles, YOLO might not detect anything (it's trained on real photos).
    -   **Action:** Download a real car image (`car.jpg`) and load it.
3.  **Experiment:**
    -   Set `conf=0.1` in `model(img, conf=0.1)`.
    -   **Result:** You will see many "Ghost" detections (False Positives).
    -   Set `conf=0.9`.
    -   **Result:** You might miss partially occluded cars (False Negatives).
    -   **Trade-off:** Precision vs Recall.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Symptom:** GPU crashes.
**Cause:** Image resolution too high or Batch size too large.
**Solution:** Resize image to 640x640. Use `yolov8n` (Nano).

#### 2. False Positives on Guardrails
**Symptom:** Guardrails detected as "Train" or "Truck".
**Cause:** Similar linear features.
**Solution:** Retrain the model on a dataset that includes guardrails as a "Background" class (Negative Mining).

---

## ⚡ Optimization & Best Practices

### 1. TensorRT
Python/PyTorch is slow for production.
-   Convert `.pt` model to ONNX.
-   Convert ONNX to **TensorRT** engine.
-   **Speedup:** 5x to 10x faster inference on NVIDIA GPUs.

### 2. Quantization (INT8)
Run the model using 8-bit integers instead of 32-bit floats.
-   **Benefit:** 4x smaller memory, 2x-4x faster.
-   **Cost:** Slight drop in accuracy (< 1%).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between Classification and Detection?
    *   **A:** Classification = "What". Detection = "What + Where".
2.  **Q:** Why do we need Anchor Boxes?
    *   **A:** To help the network learn typical object shapes (Tall for pedestrians, Wide for cars). It predicts offsets from these anchors rather than raw coordinates.
3.  **Q:** What happens if two cars are in the same grid cell?
    *   **A:** Older YOLO versions struggled. Newer versions (Anchor-free) or multiple anchors per cell handle this better.

### Challenge Task
**Task:** Video Processing.
1.  Use `cv2.VideoCapture`.
2.  Loop through frames.
3.  Run YOLO on each frame.
4.  Calculate FPS (Frames Per Second).
5.  Try to optimize to reach > 30 FPS.

---

## 📚 Further Reading & References
-   [YOLOv8 Documentation](https://docs.ultralytics.com/)
-   [Papers with Code: Object Detection](https://paperswithcode.com/task/object-detection)

---

**Day 122 Complete** | Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception
