# Day 19: Object Detection (YOLO)
## Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning

---

> **📝 Day 19 Focus:**
> Classification tells us *what* is in an image. Object Detection tells us *what* and *where*. For ADAS, this is non-negotiable. We need to know where the pedestrians, cars, and signs are in real-time. Today, we master **YOLO (You Only Look Once)**, the industry standard for fast object detection.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** between Image Classification, Object Detection, and Instance Segmentation.
2.  **Deconstruct** the YOLO architecture: Grid cells, Anchor Boxes, and the Output Tensor.
3.  **Explain** key metrics: IoU (Intersection over Union), mAP (mean Average Precision), and NMS (Non-Maximum Suppression).
4.  **Deploy** a pre-trained YOLOv8 model for real-time detection on video streams.
5.  **Fine-tune** YOLOv8 on a custom dataset (e.g., Pothole detection).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 18:** CNNs and Convolution.
-   **Python:** PyTorch.

### Hardware Requirements
-   **GPU:** Highly recommended for training. CPU is okay for inference (YOLOv8n).

### Software Stack
-   **Libraries:** `ultralytics` (YOLOv8), `opencv-python`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Evolution of Detection

#### 1.1 Two-Stage Detectors (R-CNN Family)
-   **Step 1:** Propose regions (Region Proposal Network). "There might be an object here."
-   **Step 2:** Classify regions. "It's a cat."
-   **Pros:** Accurate.
-   **Cons:** Slow (5-10 FPS). Not suitable for ADAS.

#### 1.2 One-Stage Detectors (YOLO, SSD)
-   **Concept:** Treat detection as a single regression problem.
-   **Process:** Feed image -> CNN -> Output Tensor (Bounding Boxes + Classes).
-   **Pros:** Fast (60+ FPS). Real-time.

---

### 🔹 Part 2: How YOLO Works

#### 2.1 The Grid
YOLO divides the input image (e.g., 640x640) into an $S \times S$ grid (e.g., 20x20).
-   If the center of an object falls into a grid cell, that cell is responsible for detecting it.

#### 2.2 The Output Vector
Each grid cell predicts $B$ bounding boxes. For each box, it predicts:
-   **Coordinates:** $x, y, w, h$ (Relative to cell).
-   **Objectness Score:** Probability that a box contains *an* object.
-   **Class Probabilities:** $P(Car), P(Pedestrian), \dots$

Output Tensor Size: $S \times S \times (B \times (5 + C))$.

#### 2.3 Anchor Boxes
Objects have different shapes (Pedestrians are tall/thin, Cars are wide).
Instead of predicting box shape from scratch, YOLO predicts offsets from pre-defined **Anchor Boxes** (Templates).

#### 2.4 Non-Maximum Suppression (NMS)
The network might predict 5 boxes for the same car.
**NMS Algorithm:**
1.  Sort boxes by confidence.
2.  Pick the highest confidence box ($A$).
3.  Discard any other box ($B$) if $IoU(A, B) > Threshold$ (e.g., 0.5).
4.  Repeat.

---

### 🔹 Part 3: Metrics

#### 3.1 Intersection over Union (IoU)
$$ IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}} $$
-   IoU > 0.5: Decent match.
-   IoU > 0.7: Good match.

#### 3.2 mean Average Precision (mAP)
-   **Precision:** TP / (TP + FP). (How many predicted cars are actually cars?)
-   **Recall:** TP / (TP + FN). (How many actual cars did we find?)
-   **AP:** Area under the Precision-Recall curve.
-   **mAP:** Average AP across all classes.
-   **mAP@0.5:** mAP calculated with IoU threshold 0.5.

---

## 💻 Implementation: YOLOv8 Real-Time Detection

We will use the `ultralytics` library, which makes YOLOv8 incredibly easy to use.

### 🛠️ Setup
Create `week3_day19` and install dependencies.

```bash
mkdir -p ~/ros2_ws/src/week3_day19
cd ~/ros2_ws/src/week3_day19
pip install ultralytics
touch yolo_inference.py
```

### 👨‍💻 Code: Real-Time Inference

```python
import cv2
from ultralytics import YOLO
import time

def run_yolo_webcam():
    # 1. Load Model
    # 'yolov8n.pt' is the Nano version (fastest, least accurate)
    # 'yolov8x.pt' is the Xtra Large version (slowest, most accurate)
    print("Loading Model...")
    model = YOLO('yolov8n.pt') 

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    print("Starting Inference. Press 'q' to quit.")

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret: break

        # 3. Inference
        # stream=True returns a generator for memory efficiency
        results = model(frame, stream=True, verbose=False)

        # 4. Process Results
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confidence
                conf = math.ceil((box.conf[0]*100))/100
                
                # Class Name
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} {conf}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS Calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.1f}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('YOLOv8 Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import math
if __name__ == "__main__":
    run_yolo_webcam()
```

### 👨‍💻 Code: Training on Custom Data (Snippet)

To train YOLOv8, you need a dataset in YOLO format (images and .txt labels).

```python
from ultralytics import YOLO

def train_custom():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # data='coco128.yaml' is a sample dataset. Replace with your 'data.yaml'
    results = model.train(data='coco128.yaml', epochs=10, imgsz=640, device=0)
    
    # Evaluate
    metrics = model.val()
    
    # Export
    success = model.export(format='onnx')

if __name__ == "__main__":
    # train_custom() # Uncomment to run
    pass
```

---

## 🔬 Lab Exercise: Traffic Sign Detection

### Lab Objectives
1.  **Data:** Download a small Traffic Sign dataset (or use COCO128).
2.  **Train:** Run the training script for 50 epochs.
3.  **Inference:** Test the trained model (`best.pt`) on a video of driving.
4.  **Observation:** Compare `yolov8n` vs `yolov8m`.
    -   Nano: 100+ FPS, misses small signs.
    -   Medium: 30 FPS, catches more signs.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Low FPS
**Cause:** Running on CPU or using a large model (`yolov8x`).
**Solution:**
-   Use `yolov8n`.
-   Reduce image size (`imgsz=320`).
-   Use GPU (`device=0`).

#### 2. False Positives
**Symptom:** Detecting a "Car" in the clouds.
**Cause:** Training data didn't have enough "background" images (images with no objects).
**Solution:** Add empty images to the training set.

#### 3. Missed Small Objects
**Cause:** Downsampling in CNN loses detail.
**Solution:**
-   Increase `imgsz`.
-   Use a model with P2 layer (High resolution features).

---

## ⚡ Optimization & Best Practices

### 1. TensorRT Export
For deployment on NVIDIA Jetson (Robot), export to TensorRT.
```python
model.export(format='engine') # Creates .engine file
```
This can speed up inference by 2-5x.

### 2. Quantization (INT8)
Convert weights from Float32 to Int8. Slight accuracy drop, massive speedup.

### 3. Half Precision (FP16)
Use `model.predict(..., half=True)`. Almost no accuracy loss, 2x speedup on Tensor Cores.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between Classification and Detection?
    *   **A:** Classification = "Dog". Detection = "Dog at [x, y, w, h]".
2.  **Q:** Why do we use Anchor Boxes?
    *   **A:** To help the network learn typical object shapes (priors) faster than learning from scratch.
3.  **Q:** What does NMS do?
    *   **A:** Removes overlapping duplicate boxes for the same object, keeping only the best one.

### Challenge Task
**Task:** Distance Estimation.
1.  Detect a Car.
2.  Assume the car width is 1.8 meters.
3.  Use the Pinhole model (Day 15) and the width of the bounding box (in pixels) to estimate the distance $Z$.
    $$ Z = \frac{f \times \text{Real Width}}{\text{Pixel Width}} $$
4.  Display distance on the video.

---

## 📚 Further Reading & References
-   [YOLOv8 Documentation](https://docs.ultralytics.com/)
-   [YOLOv1 Paper (Redmon et al.)](https://arxiv.org/abs/1506.02640) - The original idea.

---

**Day 19 Complete** | Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning
