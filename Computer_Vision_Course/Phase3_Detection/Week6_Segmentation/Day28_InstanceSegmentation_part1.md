# Day 28 Deep Dive: RoIAlign & Keypoint Detection

## 1. RoIAlign Mathematics
Given a RoI of size $w \times h$ on the feature map. We want to pool it to $7 \times 7$.
*   **RoIPool:** Round $w/7$ and $h/7$ to integers. This causes shifts of up to 0.5 stride pixels. With stride 32, this is 16 pixels error!
*   **RoIAlign:**
    1.  Divide RoI into $7 \times 7$ bins (floating point boundaries).
    2.  In each bin, define 4 sampling points.
    3.  Compute value at each point using Bilinear Interpolation from the grid features.
    4.  Max/Avg pool the 4 values to get the bin value.

## 2. Keypoint Detection (Human Pose)
Mask R-CNN is easily extensible.
*   **Task:** Detect 17 Keypoints (Left Eye, Right Elbow, etc.).
*   **Keypoint Head:**
    *   Similar to Mask Head.
    *   Output: 17 Heatmaps (one per keypoint).
    *   Target: One-hot mask with a single pixel set to 1 (or Gaussian blob).
    *   Loss: Cross-Entropy over the heatmap.

## 3. YOLACT (You Only Look At CoefficienTs)
**Real-time Instance Segmentation.**
*   **Idea:** Mask R-CNN is slow (Two-stage).
*   **Mechanism:**
    1.  **Prototypical Masks:** Predict $k$ prototype masks for the whole image (FCN).
    2.  **Mask Coefficients:** Predict $k$ coefficients for each object instance (Box head).
    3.  **Assembly:** Linear combination of prototypes using coefficients + Crop to box.
*   **Speed:** 30+ FPS (vs 5 FPS for Mask R-CNN).

## 4. SOLO (Segmenting Objects by Locations)
**Anchor-Free Instance Segmentation.**
*   Divides image into $S \times S$ grid.
*   If object center falls in grid $(i, j)$, that grid cell predicts the instance mask.
*   **Decoupled Head:** Predicts mask kernel weights dynamically.

## Summary
While Mask R-CNN is the accuracy king, single-stage methods like YOLACT and SOLO are pushing the boundaries of speed, making real-time instance segmentation possible.
