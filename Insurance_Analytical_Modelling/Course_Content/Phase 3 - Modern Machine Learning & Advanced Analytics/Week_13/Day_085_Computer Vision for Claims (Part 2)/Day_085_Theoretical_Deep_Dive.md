# Computer Vision for Claims (Part 2) - Theoretical Deep Dive

## Overview
Classifying an image as "Damaged" is just step one. To estimate the repair cost, we need to know *what* is damaged and *how much*. This session covers **Object Detection (YOLO)** and **Segmentation (U-Net)** for granular damage assessment.

---

## 1. Conceptual Foundation

### 1.1 Object Detection (YOLO)

*   **Goal:** Draw a box around every car part.
*   **Classes:** "Bumper", "Headlight", "Door", "Hood".
*   **YOLO (You Only Look Once):**
    *   Splits the image into a grid.
    *   Each grid cell predicts: "Is there an object center here?" + "How wide/tall is it?" + "What class is it?"
    *   *Speed:* Extremely fast (Real-time).

### 1.2 Semantic Segmentation (U-Net)

*   **Goal:** Color every pixel.
    *   Pixel (10, 10) -> "Scratch".
    *   Pixel (10, 11) -> "Paint".
*   **U-Net Architecture:**
    *   **Encoder:** Downsamples image to capture context (The "What").
    *   **Decoder:** Upsamples image to restore resolution (The "Where").
    *   **Skip Connections:** Connect Encoder to Decoder to preserve fine details (like thin scratches).

### 1.3 Instance Segmentation (Mask R-CNN)

*   **Goal:** Separate "Car A" from "Car B".
*   **Semantic Segmentation:** Treats all cars as one blob of "Car pixels".
*   **Instance Segmentation:** Gives each car a unique ID and mask.
*   *Actuarial Use:* Counting how many panels need repainting.

---

## 2. Mathematical Framework

### 2.1 IoU (Intersection over Union)

$$ \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} $$
*   Used to measure accuracy of Bounding Boxes and Masks.
*   *Threshold:* If IoU > 0.5, we count it as a correct detection.

### 2.2 YOLO Loss Function

$$ Loss = \lambda_{coord} \sum (x - \hat{x})^2 + \sum (p - \hat{p})^2 + \sum (class - \hat{class})^2 $$
*   Combines:
    1.  **Coordinate Error:** Is the box in the right place?
    2.  **Objectness Error:** Is there an object here?
    3.  **Class Error:** Is it a Bumper or a Door?

---

## 3. Theoretical Properties

### 3.1 The "Anchor Box" Concept

*   YOLO doesn't start from scratch. It assumes "priors" (Anchor Boxes).
*   It predicts *offsets* from these anchors.
*   *Insight:* Most car parts have standard aspect ratios (Bumpers are wide, Doors are tall). Anchors help the model learn faster.

### 3.2 Automated Cost Estimation

*   **Logic:**
    1.  Detect "Front Bumper" (Confidence 0.9).
    2.  Segment "Scratch" on "Front Bumper" (Area: 10% of bumper).
    3.  Lookup Repair Rules: "If scratch < 20%, Repair. If > 20%, Replace."
    4.  Estimate: $200 (Repair Labor).

---

## 4. Modeling Artifacts & Implementation

### 4.1 YOLOv8 (Ultralytics)

```python
from ultralytics import YOLO

# 1. Load Pre-trained Model (COCO dataset)
model = YOLO('yolov8n.pt') 

# 2. Train on Car Parts Dataset
# model.train(data='car_parts.yaml', epochs=100)

# 3. Inference
results = model('crash_scene.jpg')

# 4. Process Results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected {model.names[cls]} with confidence {conf:.2f}")
```

### 4.2 U-Net (PyTorch)

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(3, 64)
        self.pool = nn.MaxPool2d(2)
        # ... more layers ...
        
        # Decoder (Upsampling)
        self.up = nn.Upsample(scale_factor=2)
        self.dec1 = self.conv_block(128, 64) # 128 due to skip connection
        
    def forward(self, x):
        e1 = self.enc1(x)
        x = self.pool(e1)
        # ... bottleneck ...
        x = self.up(x)
        x = torch.cat([x, e1], dim=1) # Skip Connection
        x = self.dec1(x)
        return x # Output: Mask
```

---

## 5. Evaluation & Validation

### 5.1 mAP (Mean Average Precision)

*   The gold standard metric for Object Detection.
*   Calculates Precision at different Recall levels and averages them.
*   *Actuarial Standard:* mAP@0.5 > 0.8 is usually required for production.

### 5.2 The "Dent" vs. "Reflection" Problem

*   Shiny cars reflect the sky/trees.
*   Models often confuse reflections with dents.
*   *Fix:* Train on data with diverse lighting conditions. Use "Motion Parallax" (Video) if available.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Small Object Detection**
    *   YOLO struggles with tiny objects (e.g., a hail dent from far away).
    *   *Fix:* Use "Tiling" (Crop image into smaller squares and run detection on each).

2.  **Trap: Class Imbalance**
    *   "Bumper" appears in every photo. "Roof Rack" is rare.
    *   *Fix:* Oversampling or Focal Loss.

### 6.2 Implementation Challenges

1.  **Labeling Data:**
    *   Drawing bounding boxes is fast.
    *   Drawing segmentation masks (tracing scratches) is slow and expensive.
    *   *Fix:* Use "SAM" (Segment Anything Model) to assist annotators.

---

## 7. Advanced Topics & Extensions

### 7.1 SAM (Segment Anything Model)

*   A foundation model from Meta.
*   Can segment any object given a simple prompt (click or box).
*   *Use Case:* Interactive labeling for adjusters.

### 7.2 3D Reconstruction (NeRF)

*   Take 5 photos around the car.
*   Reconstruct a 3D mesh.
*   Measure the *depth* of the dent (crucial for repair cost).

---

## 8. Regulatory & Governance Considerations

### 8.1 "Right to Repair"

*   If the AI says "Repair" but the shop says "Replace", who wins?
*   **Governance:** The AI is a "Recommendation Engine". The human adjuster has the final override.

---

## 9. Practical Example

### 9.1 Worked Example: The "Hail" Catastrophe

**Scenario:**
*   Hailstorm hits a dealership. 500 cars damaged.
*   **Manual:** 20 mins per car = 166 hours.
*   **Drone + Computer Vision:**
    *   Drone flies over lot.
    *   YOLO detects "Hail Dents".
    *   Count dents per hood/roof.
*   **Result:** Assessed all 500 cars in 2 hours. Payouts issued next day.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **YOLO** finds the parts.
2.  **U-Net** finds the damage pixels.
3.  **IoU** measures success.

### 10.2 When to Use This Knowledge
*   **Auto Claims:** The primary use case.
*   **Property:** Detecting roof damage from satellite.

### 10.3 Critical Success Factors
1.  **High-Quality Annotations:** The model is only as good as the masks it trained on.
2.  **Lighting:** Train on sunny, cloudy, and rainy days.

### 10.4 Further Reading
*   **Redmon et al.:** "You Only Look Once: Unified, Real-Time Object Detection".
*   **Ronneberger et al.:** "U-Net: Convolutional Networks for Biomedical Image Segmentation".

---

## Appendix

### A. Glossary
*   **Bounding Box:** Rectangle defined by $(x, y, w, h)$.
*   **Mask:** Binary image (0=Background, 1=Object).
*   **Inference:** Running the model on new data.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **IoU** | $A \cap B / A \cup B$ | Accuracy Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
