# Day 43: Deep Learning for Vision (CNNs & YOLO)
## Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI

---

## 🎯 Learning Objectives
1.  **Understand** Convolutional Neural Networks (CNNs) and why they outperform classical CV.
2.  **Analyze** Object Detection architectures: YOLO (You Only Look Once) vs SSD (Single Shot Detector).
3.  **Run** pre-trained models using OpenCV DNN module.
4.  **Interpret** model outputs (Bounding Boxes, Class IDs, Confidence Scores).
5.  **Perform** Non-Maximum Suppression (NMS) to clean up detections.
6.  **Optimize** inference for speed (Input resolution, Quantization).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** PC with GPU (recommended) or CPU.
*   **Software:** OpenCV 4.x (DNN module), Model Weights (`yolov3.weights`, `yolov3.cfg`, `coco.names`).
*   **Knowledge:** Neural Networks basics (Layers, Weights, Activation).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: From Features to Learning
*   **Classical CV (Day 41/42):** We hand-crafted features (Corners, HOG).
*   **Deep Learning:** The network *learns* the best features from data.
*   **CNN:** Uses convolution layers to extract hierarchical features (Edges -> Shapes -> Objects).

### 🔹 Part 2: Object Detection Architectures
*   **Classification:** "Is there a cat?" (Output: Class).
*   **Detection:** "Where is the cat?" (Output: Class + Bounding Box).
*   **Two-Stage (R-CNN):** Propose regions -> Classify regions. Accurate but slow.
*   **One-Stage (YOLO/SSD):** Predict boxes and classes in a single pass. Fast.
    *   **YOLO:** Divides image into a grid. Each cell predicts B boxes and C class probabilities.
    *   **SSD:** Uses multi-scale feature maps to detect objects of different sizes.

### 🔹 Part 3: The Output Format
A typical YOLO output vector for one bounding box:
`[Center_X, Center_Y, Width, Height, Confidence, Class1_Prob, Class2_Prob, ...]`
*   **Confidence:** Objectness score (Is there an object?).
*   **Class Prob:** Conditional probability (If object, what is it?).

---

## 💻 Implementation Examples

### Example 1: Running YOLOv3 with OpenCV

```cpp
/**
 * @brief YOLO Object Detection
 */
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>

using namespace cv;
using namespace cv::dnn;

void detect_yolo(Mat& img) {
    // 1. Load Model
    Net net = readNetFromDarknet("yolov3.cfg", "yolov3.weights");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    // 2. Prepare Input Blob
    // YOLO expects 416x416, scaled by 1/255
    Mat blob;
    blobFromImage(img, blob, 1/255.0, Size(416, 416), Scalar(0,0,0), true, false);
    net.setInput(blob);
    
    // 3. Forward Pass
    std::vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    
    // 4. Post-Process
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    
    for (auto& out : outs) {
        float* data = (float*)out.data;
        for (int j = 0; j < out.rows; ++j, data += out.cols) {
            Mat scores = out.row(j).colRange(5, out.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            
            if (confidence > 0.5) {
                int centerX = (int)(data[0] * img.cols);
                int centerY = (int)(data[1] * img.rows);
                int width = (int)(data[2] * img.cols);
                int height = (int)(data[3] * img.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // 5. NMS (Non-Maximum Suppression)
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
    
    // 6. Draw
    for (int idx : indices) {
        rectangle(img, boxes[idx], Scalar(0, 255, 0), 2);
        // Draw Label...
    }
}
```

### Example 2: Loading Class Names

```cpp
std::vector<std::string> load_classes(std::string filename) {
    std::vector<std::string> classes;
    std::ifstream ifs(filename.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);
    return classes;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: YOLO-Tiny Speed Test

**Objective:** Compare standard YOLO vs YOLO-Tiny.

**Steps:**
1.  Run `yolov3.weights` (Standard). Measure Inference Time (e.g., 200ms on CPU).
2.  Run `yolov3-tiny.weights` (Tiny). Measure Inference Time (e.g., 30ms on CPU).
3.  **Observation:** Tiny is much faster but less accurate (misses small objects).
4.  **Use Case:** Tiny is perfect for Raspberry Pi / Embedded.

### Lab 2: Confidence Thresholding

**Objective:** Filter garbage detections.

**Steps:**
1.  Set Confidence Threshold = 0.1.
2.  **Result:** Many boxes, lots of noise (detecting clouds as sheep).
3.  Set Confidence Threshold = 0.9.
4.  **Result:** Very few boxes, only the most obvious objects.
5.  **Sweet Spot:** Usually 0.5.

### Lab 3: NMS Tuning

**Objective:** Understand Non-Maximum Suppression.

**Steps:**
1.  Disable NMS.
2.  **Result:** You see 10 overlapping boxes around a single car.
3.  Enable NMS with IoU (Intersection over Union) Threshold = 0.4.
4.  **Result:** Only the single best box remains.

---

## 🐛 Debugging Techniques

### Debug 1: Wrong Bounding Boxes

**Symptom:** Boxes are offset or wrong size.

**Cause:**
*   Input Blob scaling wrong. YOLO expects 0-1 range (scale 1/255). If you pass 0-255, it fails.
*   BGR vs RGB mismatch (`swapRB` parameter in `blobFromImage`).
*   **Fix:** Check `blobFromImage` parameters carefully.

### Debug 2: Slow Inference

**Symptom:** 1 FPS.

**Cause:**
*   Running on CPU without optimization.
*   Input resolution too high (e.g., 1920x1080).
*   **Fix:** Resize input to 416x416 or 320x320. Use OpenVINO or CUDA backend if available.

---

## ⚡ Performance Optimization

### Optimization 1: Input Resolution

*   YOLO is fully convolutional, so it accepts any size.
*   320x320: Fast, lower accuracy.
*   608x608: Slow, detects small objects better.
*   **Trade-off:** Choose the smallest size that still detects your target.

### Optimization 2: DNN Backends

*   `net.setPreferableBackend(DNN_BACKEND_CUDA)`: Uses Nvidia GPU.
*   `net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE)`: Uses Intel OpenVINO (CPU/iGPU/VPU).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Intersection over Union" (IoU)?**
2.  **Why do we need "Anchor Boxes" in YOLO?**
3.  **What is the difference between "Objectness" and "Class Probability"?**
4.  **Why is YOLO faster than R-CNN?**

### Practical Challenges

1.  **Build a "People Counter":** Draw a line in the center of the frame. Count how many bounding boxes cross the line.
2.  **Implement "Social Distancing Detector":** Detect people. Calculate distance between centers (in pixels). If distance < Threshold, draw box in Red.

---

## 📚 Further Reading & Resources

### Models
*   **Darknet Model Zoo:** Source for YOLO weights.
*   **ONNX Model Zoo:** Standard format for interoperability.

### Papers
*   **"YOLOv3: An Incremental Improvement"** - Redmon & Farhadi.

---

## 🎓 Summary

Today we covered:
- ✅ **Deep Learning:** The modern approach.
- ✅ **YOLO:** Fast, single-stage detection.
- ✅ **OpenCV DNN:** Loading and running models.
- ✅ **Post-Processing:** NMS and Thresholding.
- ✅ **Trade-offs:** Speed vs Accuracy (Tiny vs Full).

**Next:** Day 44 - Edge AI Acceleration (TensorRT & TFLite).

---

**Day 43 Complete** | Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI
