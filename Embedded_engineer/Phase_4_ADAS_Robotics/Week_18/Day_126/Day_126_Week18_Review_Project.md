# Day 126: Week 18 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception

---

> **📝 Day 126 Focus:**
> We have built the eyes of the car. We can classify signs (CNN), detect cars in 2D (YOLO), find lanes (UNet), and see in 3D (PointPillars). Today, we combine them into a **Perception Stack** that feeds the Planning system.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Architect** a multi-modal perception pipeline.
2.  **Synchronize** Camera and Lidar streams.
3.  **Run** multiple Neural Networks in parallel (Model Serving).
4.  **Visualize** the combined output (3D Boxes + Lane Masks).
5.  **Evaluate** the latency of the full stack.

---

## 📚 Week 18 Review

### 1. Neural Networks
-   **MLP:** Good for simple logic.
-   **CNN:** Good for images (Spatial features).
-   **Backprop:** The learning algorithm.

### 2. 2D Vision
-   **Classification:** ResNet/VGG. "What is it?"
-   **Detection:** YOLO/SSD. "Where is it (Box)?"
-   **Segmentation:** UNet. "Where is it (Pixel)?"

### 3. 3D Vision
-   **Point Clouds:** Sparse, unordered.
-   **PointPillars:** Convert PC to BEV Image -> 2D CNN.
-   **BEV Fusion:** Lift Camera to 3D + Fuse with Lidar.

---

## 🛠️ Capstone Project: The Perception Stack

**Goal:** Process a frame of data (Image + Point Cloud).
**Pipeline:**
1.  **Input:** Image ($640 \times 480$), PC ($N \times 4$).
2.  **Task 1 (Traffic Signs):** Crop ROIs -> CNN Classifier.
3.  **Task 2 (Lanes):** UNet Segmentation.
4.  **Task 3 (3D Objects):** PointPillars.
5.  **Output:** `PerceptionFrame` object containing all results.

### Package Structure
Create `week18_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week18_project
cd ~/ros2_ws/src/week18_project
touch perception_stack.py
```

### 👨‍💻 Code: The Full Stack

```python
import time
import numpy as np
import torch
import cv2

# --- Mock Models (Placeholders for real weights) ---
class MockYOLO:
    def __call__(self, img):
        # Return dummy boxes [x1, y1, x2, y2, conf, cls]
        return np.array([[100, 100, 200, 200, 0.9, 0]])

class MockUNet:
    def __call__(self, img):
        # Return dummy mask
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

class MockPointPillars:
    def __call__(self, points):
        # Return dummy 3D boxes [x, y, z, l, w, h, theta]
        return np.array([[10, 5, 0, 4, 2, 1.5, 0]])

class PerceptionStack:
    def __init__(self):
        print("Initializing Models...")
        self.yolo = MockYOLO()
        self.unet = MockUNet()
        self.pp = MockPointPillars()
        
        # In real life, load weights here
        # self.yolo = torch.hub.load(...)
        
    def process_frame(self, image, points):
        start_time = time.time()
        
        # 1. 2D Detection (YOLO)
        boxes_2d = self.yolo(image)
        
        # 2. Lane Segmentation (UNet)
        # Resize for speed
        img_small = cv2.resize(image, (256, 256))
        lane_mask = self.unet(img_small)
        
        # 3. 3D Detection (PointPillars)
        boxes_3d = self.pp(points)
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'boxes_2d': boxes_2d,
            'lane_mask': lane_mask,
            'boxes_3d': boxes_3d,
            'latency_ms': latency
        }

def visualize(image, result):
    # Draw 2D Boxes
    for box in result['boxes_2d']:
        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # Draw Latency
    cv2.putText(image, f"Latency: {result['latency_ms']:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    # Show (In real lab)
    # cv2.imshow("Perception", image)
    # cv2.waitKey(0)
    print("Visualization Complete.")

def main():
    stack = PerceptionStack()
    
    # Dummy Data
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    points = np.random.rand(1000, 4)
    
    print("Processing Stream...")
    for i in range(10):
        result = stack.process_frame(img, points)
        print(f"Frame {i}: Found {len(result['boxes_2d'])} 2D objs, {len(result['boxes_3d'])} 3D objs. Time: {result['latency_ms']:.1f}ms")
        
    visualize(img, result)

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. Latency Budget
-   **Goal:** Total latency < 100ms (10 Hz).
-   **Observation:** Even with mock models, Python overhead exists.
-   **Optimization:**
    -   Run models in parallel threads? (Python GIL prevents this for CPU, but GPU calls are async).
    -   Use `multiprocessing`.
    -   Use **TensorRT** (C++ Inference).

### 2. Synchronization
-   **Scenario:** Camera is at $t=1.00$, Lidar is at $t=1.05$.
-   **Problem:** Fast moving car moves 1.5m in 50ms.
-   **Solution:** Motion Compensation. Project Lidar points to $t=1.00$ using Ego-Motion (Odometry).

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Theory
1.  **Q:** Why do we need both 2D and 3D detection?
    *   **A:** 2D is better for classification (Traffic Lights colors). 3D is better for location (Distance to car).
2.  **Q:** What is the input to PointPillars?
    *   **A:** Raw Point Cloud $(x, y, z, i)$.

### Section 2: Implementation
3.  **Q:** How do we speed up UNet?
    *   **A:** Reduce input resolution ($1024 \to 256$) or use a lighter backbone (MobileNet instead of ResNet).
4.  **Q:** What format is the output of YOLO?
    *   **A:** `[x_center, y_center, width, height, confidence, class_id]`.

---

## 🏆 Conclusion

Congratulations on completing Week 18!
-   You have mastered **Deep Learning for Perception**.
-   You can build the eyes of an autonomous vehicle.

**Next Week:** We teach the car to *think*. **Localization & SLAM**. Where am I? And where is the map?

---

**Day 126 Complete** | Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception
