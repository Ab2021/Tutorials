# Day 24: EfficientDet & Scalable Detection

## 1. The Goal: Efficiency
**Problem:** Most detectors (RetinaNet, Faster R-CNN) are either accurate but slow, or fast but inaccurate.
**Solution:** **EfficientDet (2020)**.
*   Scalable architecture that achieves SOTA accuracy with much fewer FLOPs.

## 2. Architecture Components

### A. EfficientNet Backbone
Uses EfficientNet (B0-B7) as the feature extractor.
*   Provides high-quality features at various scales.

### B. BiFPN (Bidirectional Feature Pyramid Network)
**Evolution of FPN:**
1.  **FPN:** Top-down path only.
2.  **PANet:** Top-down + Bottom-up path.
3.  **BiFPN:**
    *   Removes nodes with only one input (simplified).
    *   Adds extra edge from input to output node (skip connection).
    *   **Weighted Feature Fusion:** Not all features are equal. Learn weights $w_i$ to combine them: $O = \sum \frac{w_i}{\epsilon + \sum w_j} \cdot I_i$.

### C. Compound Scaling
Just like EfficientNet, EfficientDet scales all dimensions jointly:
*   **Backbone:** EfficientNet B0 $\to$ B7.
*   **BiFPN:** Increase depth (layers) and width (channels).
*   **Box/Class Network:** Increase depth.
*   **Resolution:** Increase input size ($512 \to 1536$).

## 3. Comparison
| Model | AP | Params | FLOPs |
| :--- | :--- | :--- | :--- |
| YOLOv3 | 33.0 | 62M | 156B |
| RetinaNet | 40.8 | 38M | 239B |
| **EfficientDet-D0** | 34.6 | **4M** | **2.5B** |
| **EfficientDet-D7** | 55.1 | 52M | 325B |

## Summary
EfficientDet demonstrates that careful architecture design (BiFPN) and principled scaling (Compound Scaling) can yield massive efficiency gains over hand-crafted architectures.
