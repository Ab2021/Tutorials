# Day 28: Instance Segmentation (Mask R-CNN)

## 1. Problem Definition
**Goal:** Detect objects AND segment their masks.
*   **Input:** Image.
*   **Output:** Bounding Box + Class Label + Binary Mask for each object instance.
*   **Difference from Semantic:** Distinguishes "Person 1" from "Person 2".

## 2. Mask R-CNN (2017)
**Idea:** Extend Faster R-CNN by adding a branch for predicting an object mask.
*   **Backbone:** ResNet-FPN.
*   **RPN:** Generates proposals.
*   **RoIAlign:** Extracts features for each proposal (Crucial improvement over RoIPool).
*   **Heads:**
    1.  **Class/Box Head:** Predicts class and box offsets (FC layers).
    2.  **Mask Head:** Predicts binary mask (FCN).

## 3. RoIAlign vs RoIPool
**RoIPool (Faster R-CNN):**
*   Quantizes the RoI boundaries to integers (e.g., $20.5 \to 20$).
*   Divides into bins and quantizes again.
*   **Issue:** Misalignment between feature map and original image. OK for classification, bad for pixel-perfect masks.

**RoIAlign (Mask R-CNN):**
*   No quantization.
*   Uses **Bilinear Interpolation** to compute exact values of features at 4 sampling points in each bin.
*   **Result:** Pixel-accurate alignment.

## 4. Mask Head Architecture
*   Input: $14 \times 14$ feature map from RoIAlign.
*   Layers: 4 Conv layers ($3 \times 3$, 256 channels).
*   Upsample: Transposed Conv ($2 \times 2$) to $28 \times 28$.
*   Output: $1 \times 1$ Conv to $K$ classes (Sigmoid).
*   **Loss:** Binary Cross-Entropy per pixel (only for the ground truth class).

```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load pre-trained model
model = maskrcnn_resnet50_fpn(pretrained=True)

# Inference
model.eval()
x = [torch.rand(3, 300, 400)]
predictions = model(x)

# Output format
# predictions[0]['boxes']  (N, 4)
# predictions[0]['labels'] (N)
# predictions[0]['scores'] (N)
# predictions[0]['masks']  (N, 1, H, W)
```

## Summary
Mask R-CNN is a simple, flexible, and general framework for instance segmentation. By adding a small mask branch to Faster R-CNN and fixing alignment with RoIAlign, it achieved SOTA results.
