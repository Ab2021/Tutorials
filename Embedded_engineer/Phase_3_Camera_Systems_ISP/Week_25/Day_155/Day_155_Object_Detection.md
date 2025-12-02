# Day 155: Object Detection (YOLO)
## Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems

---

## 🎯 Learning Objectives
1.  **Understand** the YOLO (You Only Look Once) architecture: Grid, Anchors, IoU.
2.  **Deploy** YOLOv8/v9 on an embedded device.
3.  **Interpret** the output: Bounding Boxes (xywh), Confidence, Class ID.
4.  **Implement** NMS (Non-Maximum Suppression) to remove duplicate detections.
5.  **Visualize** detections on a live video stream.

---

## 📚 Prerequisites & Preparation
*   **Software:** Ultralytics YOLO (`pip install ultralytics`), OpenCV.
*   **Model:** `yolov8n.pt` (Nano version for Edge).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How YOLO Works
*   **Grid:** Splits the image into $S \times S$ cells.
*   **Prediction:** Each cell predicts $B$ bounding boxes and $C$ class probabilities.
*   **Anchor Boxes:** Pre-defined shapes (tall, wide) to help the network regress box dimensions.
*   **Single Shot:** Unlike R-CNN (which has a Region Proposal step), YOLO does everything in one forward pass. Fast!

### 🔹 Part 2: The Output Tensor
*   Shape: $(Batch, 84, 8400)$ for YOLOv8.
    *   $84 = 4 (x, y, w, h) + 80 (Classes)$.
    *   $8400 =$ Total number of predictions.
*   **Post-Processing:** We need to filter these 8400 boxes to find the 5 real objects.

### 🔹 Part 3: Non-Maximum Suppression (NMS)
1.  Discard boxes with Confidence < Threshold (e.g., 0.25).
2.  Sort remaining boxes by confidence.
3.  Pick the best box.
4.  Discard all other boxes that overlap (IoU > 0.45) with the best box.
5.  Repeat.

---

## 💻 Implementation Examples

### Example 1: Running YOLOv8 (Python)

The easy way.

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Open Camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Inference
    results = model(frame, verbose=False)
    
    # Plot
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLOv8", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 2: Manual Post-Processing (Understanding the Tensor)

If you use TensorRT, you get raw tensors. You must implement NMS manually.

```python
import numpy as np

def nms(boxes, scores, iou_threshold):
    # boxes: [x1, y1, x2, y2]
    # scores: [confidence]
    
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1: break
        
        # Calculate IoU with the rest
        rest = indices[1:]
        ious = compute_iou(boxes[current], boxes[rest])
        
        # Keep only those with IoU < Threshold
        indices = rest[ious < iou_threshold]
        
    return keep

def compute_iou(box, boxes):
    # Intersection over Union logic...
    # (Implementation omitted for brevity)
    return iou_array
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Benchmark Nano vs Medium

**Objective:** Accuracy vs Speed trade-off.

**Steps:**
1.  Run `yolov8n.pt` (Nano). Measure FPS. Note if it detects small objects.
2.  Run `yolov8m.pt` (Medium). Measure FPS. Note the improvement in detection.
3.  **Decision:** On a Jetson Nano, you might be forced to use Nano or Small. On Orin, you can use Medium.

### Lab 2: Custom Classes

**Objective:** Filter the output.

**Steps:**
1.  Modify the code to *only* draw boxes for "Person" (Class ID 0) and "Car" (Class ID 2).
2.  Ignore "Potted Plant" or "Tie".
3.  **Application:** A security camera only cares about people.

### Lab 3: Distance Estimation (Simple)

**Objective:** How far is the person?

**Steps:**
1.  Assume average human height is 1.7m.
2.  Get the bounding box height $h$ (pixels).
3.  Use Pinhole formula: $Z = (f \times 1.7) / h$.
4.  Display the distance on top of the box.

---

## 🐛 Debugging YOLO

### Debug 1: "False Positives"

**Symptom:** Detecting a "Person" in a tree shadow.

**Cause:**
*   Training data bias.
*   Confidence threshold too low.
*   **Fix:** Increase `conf` threshold (e.g., 0.5). Or collect "Negative Samples" (images of trees) and retrain.

### Debug 2: "Flickering" Boxes

**Symptom:** Box appears and disappears rapidly.

**Cause:**
*   Borderline confidence.
*   **Fix:** Use a **Tracker** (ByteTrack/DeepSort) to smooth the detections over time. If a box is missed for 1 frame, the tracker keeps it alive.

---

## ⚡ Performance Optimization

### Optimization 1: Letterboxing

*   YOLO expects square input (640x640).
*   Camera is 1920x1080 (16:9).
*   **Don't Stretch:** Stretching distorts objects.
*   **Letterbox:** Resize to 640x360 and add gray padding to top/bottom to make it 640x640.
*   **Efficiency:** Some implementations allow rectangular inference (640x384) to save computation on padding pixels.

### Optimization 2: INT8 Calibration

*   YOLO is robust to quantization.
*   Using INT8 on Jetson Orin can yield 100+ FPS for YOLOv8n.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "IoU" (Intersection over Union)?** (Area of Overlap / Area of Union. Metric for how well two boxes align).
2.  **Why does YOLO struggle with small objects?** (Because the grid becomes coarse. A 640x640 image becomes a 20x20 grid. Small objects might vanish in the downsampling).
3.  **Difference between Object Detection and Image Classification?** (Classification = "There is a cat". Detection = "The cat is HERE (x,y,w,h)").

### Practical Challenges

1.  **Count People:** Create a "Counter" that increments when a person crosses a line (Virtual Tripwire).
2.  **Night Vision:** Test YOLO on IR camera footage. Does it work? (Usually yes, shape is preserved).

---

## 📚 Further Reading & Resources

### Documentation
*   **Ultralytics YOLOv8 Docs.**
*   **"Object Detection in 20 Years: A Survey".**

---

## 🎓 Summary

Today we covered:
- ✅ **YOLO:** The standard for speed.
- ✅ **NMS:** Cleaning up the mess.
- ✅ **Classes:** Filtering what we see.
- ✅ **Trade-offs:** Nano vs Large.
- ✅ **Letterboxing:** Handling aspect ratios.

**Next:** Day 156 - Semantic Segmentation (UNet).

---

**Day 155 Complete** | Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems


