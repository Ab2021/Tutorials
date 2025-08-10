# Day 22.3: Single-Stage Detection (YOLO, SSD) - A Practical Guide

## Introduction: Detection as a Regression Problem

While two-stage detectors like Faster R-CNN achieve high accuracy by first proposing regions and then classifying them, this two-step process can be slow. **Single-stage detectors** revolutionized the field by framing object detection as a single, unified regression problem. They skip the explicit region proposal step and instead predict bounding boxes and class probabilities directly from the full image in a single pass.

This approach, pioneered by models like **YOLO (You Only Look Once)** and **SSD (Single Shot MultiBox Detector)**, leads to dramatically faster inference speeds, making real-time object detection possible.

This guide provides a practical overview of the core ideas behind single-stage detectors.

**Today's Learning Objectives:**

1.  **Understand the Single-Stage Paradigm:** Grasp the core idea of treating object detection as a direct regression from image features.
2.  **Explore the YOLO Architecture:** Learn how YOLO divides an image into a grid and predicts boxes and class probabilities for each grid cell.
3.  **Learn about the SSD Architecture:** Understand how SSD improves on YOLO by making predictions at multiple feature maps of different scales.
4.  **Grasp the Role of Default Boxes (Anchors):** See how both YOLO and SSD use pre-defined anchor boxes to guide the prediction process.
5.  **Understand Non-Max Suppression (NMS):** Learn about this crucial post-processing step used to clean up the thousands of raw predictions from a single-stage detector.

---

## Part 1: YOLO (You Only Look Once)

YOLO was a groundbreaking paper that introduced a completely new approach to detection.

**The Core Idea:**
1.  **Grid System:** The input image is resized to a fixed size (e.g., 448x448) and divided into an `S x S` grid (e.g., 7x7).
2.  **Grid Cell Responsibility:** If the center of an object falls into a particular grid cell, that grid cell is responsible for detecting that object.
3.  **Direct Prediction:** The image is passed through a single, large CNN. The output is a single tensor of shape `(S, S, B*5 + C)`.
    *   For each of the `S x S` grid cells, the network predicts:
        *   `B` bounding boxes (e.g., `B=2`). Each box prediction consists of 5 values: `(x, y, w, h, confidence)`.
            *   `(x, y, w, h)` are the box coordinates.
            *   `confidence` is the model's confidence that this box contains an object (objectness score).
        *   `C` class probabilities, conditional on an object being present.

**The Result:** A single forward pass produces a fixed grid of predictions across the entire image.

![YOLO Grid](https://i.imgur.com/l4a4y7K.png)

**Limitation of early YOLO:** It struggled with detecting small objects, as each grid cell could only predict one class, making it difficult to detect multiple small objects that fall into the same cell.

---

## Part 2: SSD (Single Shot MultiBox Detector)

SSD built upon the ideas of YOLO and significantly improved its performance, especially for small objects.

**The Key Innovation: Multi-Scale Feature Maps**
*   **The Problem:** YOLO only makes predictions from the final feature map, which has a low spatial resolution, making it hard to detect small objects.
*   **The SSD Solution:** Instead of just using the final layer, SSD makes predictions at **multiple stages** of the CNN backbone. It uses feature maps from both deep layers (which have rich semantic information but are low-resolution) and earlier layers (which are high-resolution but have less semantic information).

**How it Works:**
1.  A standard CNN backbone (like VGG-16) is used.
2.  A set of auxiliary convolutional layers are added on top, which progressively decrease in size.
3.  A prediction head is attached to several of these feature maps, from early in the network to the very end.
4.  The prediction head at the **high-resolution (early) feature maps** is responsible for detecting **small objects**.
5.  The prediction head at the **low-resolution (deep) feature maps** is responsible for detecting **large objects**.
6.  Like YOLO, each head uses a set of default (anchor) boxes of different aspect ratios.

![SSD Architecture](https://i.imgur.com/3gQ5z0A.png)

---

## Part 3: Non-Max Suppression (NMS)

**The Problem:** Single-stage detectors produce a massive number of raw predictions. A single object might be detected by multiple grid cells or multiple anchor boxes, resulting in many overlapping bounding boxes.

**The Solution: Non-Max Suppression (NMS)**

NMS is a simple but essential post-processing algorithm to clean up these redundant detections.

**The Process:**
1.  Take the list of all predicted boxes for a given class.
2.  Discard all boxes with a confidence score below a certain threshold (e.g., 0.5).
3.  While there are still boxes left in the list:
    a. Select the box with the highest confidence score and add it to your final list of predictions.
    b. Remove this box from the initial list.
    c. For all remaining boxes in the list, calculate their IoU with the box you just selected.
    d. Discard all boxes that have an IoU above a certain threshold (e.g., 0.5), as they are likely detecting the same object.
4.  Repeat until the initial list is empty.

### 3.1. Implementation with `torchvision`

`torchvision.ops.nms` provides an efficient, vectorized implementation of NMS.

```python
import torch
import torchvision

print("--- Part 3: Non-Max Suppression (NMS) ---")

# --- Dummy Predictions ---
# Let's imagine our model predicted these boxes for the 'car' class
# Each box is [x_min, y_min, x_max, y_max]
boxes = torch.tensor([
    [100, 100, 210, 210], # High confidence, correct box
    [110, 110, 220, 220], # High confidence, highly overlapping box
    [10, 10, 50, 50],     # A separate, low-confidence box
    [150, 150, 250, 250]  # A separate, high-confidence box
], dtype=torch.float)

# Corresponding confidence scores
scores = torch.tensor([0.95, 0.90, 0.3, 0.92], dtype=torch.float)

# --- Apply NMS ---
# We set the IoU threshold to 0.5. Any box with an IoU > 0.5 with a higher-scoring
# box will be suppressed.
iou_threshold = 0.5
indices_to_keep = torchvision.ops.nms(boxes, scores, iou_threshold)

print(f"Original number of boxes: {len(boxes)}")
print(f"Indices of boxes to keep after NMS: {indices_to_keep}")

final_boxes = boxes[indices_to_keep]
final_scores = scores[indices_to_keep]

print(f"\nFinal boxes after NMS:\n{final_boxes}")
print(f"Final scores after NMS:\n{final_scores}")

# The box at index 1 was suppressed because its IoU with the higher-scoring box at index 0 was too high.
# The box at index 2 was suppressed because its score was below the internal confidence threshold of NMS.
```

---

## Part 4: Using a Pre-trained SSD in PyTorch

Let's use a pre-trained SSD model from `torchvision` to see a single-stage detector in action.

```python
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import requests

print("\n--- Part 4: Using a Pre-trained SSD ---")

# --- 1. Load the Pre-trained Model ---
# We load an SSD300 model with a VGG16 backbone.
model = torchvision.models.detection.ssd300_vgg16(weights='DEFAULT')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- 2. Load and Transform a Sample Image ---
url = 'https://www.autocar.co.uk/sites/autocar.co.uk/files/styles/gallery_full/public/volkswagen_id_buzz_01.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image).to(device)
input_batch = [img_tensor]

# --- 3. Perform Inference ---
with torch.no_grad():
    predictions = model(input_batch)

pred = predictions[0]

# --- 4. Visualize the Results ---
# (We can reuse the same visualization function from the Faster R-CNN guide)
def draw_predictions(image, predictions, threshold=0.5):
    # (Implementation from previous guide)
    pass

print("SSD model inference complete. The output format is the same as Faster R-CNN.")
print("It's a list of dictionaries containing 'boxes', 'labels', and 'scores'.")
```

## Conclusion

Single-stage detectors like YOLO and SSD fundamentally changed the field of object detection by framing it as a direct regression problem. By removing the dedicated region proposal stage, they achieved massive gains in speed, enabling real-time detection on standard hardware.

**Key Takeaways:**

1.  **Detection as Regression:** The core idea is to directly predict box coordinates and class probabilities from feature maps in a single pass.
2.  **Grid-Based Approach:** The image is divided into a grid, and each cell (or feature map location) is responsible for predicting objects whose centers fall within it.
3.  **Multi-Scale Predictions (SSD):** To improve performance on objects of various sizes, it's crucial to make predictions from multiple feature maps at different scales.
4.  **Anchors/Default Boxes:** Like in two-stage detectors, using pre-defined boxes simplifies the learning problem.
5.  **NMS is Essential:** A post-processing step like Non-Max Suppression is required to clean up the thousands of raw, overlapping predictions that these models produce.

While two-stage detectors may still hold a slight edge in accuracy for certain benchmarks, the speed and simplicity of single-stage detectors have made them the go-to choice for a vast range of real-time applications.

## Self-Assessment Questions

1.  **Single-Stage vs. Two-Stage:** What is the main architectural component of a two-stage detector that is completely absent in a single-stage detector?
2.  **YOLO Grid:** In the original YOLO, if the center of two small objects falls into the same grid cell, what problem does this create?
3.  **SSD Multi-Scale:** In SSD, which feature maps are responsible for detecting small objects: the early, high-resolution maps or the deep, low-resolution maps?
4.  **NMS:** What are the two criteria used by Non-Max Suppression to filter out a bounding box?
5.  **Use Case:** You are building a system to detect objects from a live video feed from a drone, where inference speed is the absolute top priority. Would you choose Faster R-CNN or YOLO? Why?
