# Day 23 Deep Dive: FCOS & CenterNet Details

## 1. FCOS Multi-Level Prediction
How to handle overlapping objects?
*   **Ambiguity:** If a pixel falls inside two boxes, which one does it predict?
*   **Solution:** Feature Pyramid Network (FPN).
    *   Assign objects to different FPN levels based on size.
    *   Small objects $\to$ High-res feature map (P3).
    *   Large objects $\to$ Low-res feature map (P7).
    *   Most overlaps are resolved because objects of different sizes are handled at different levels.

## 2. CenterNet Heatmap Generation
How do we create the Ground Truth heatmap?
*   For each object center $(p_x, p_y)$, splat a Gaussian kernel onto the heatmap.
    $$ Y_{xy} = \exp\left(-\frac{(x-p_x)^2 + (y-p_y)^2}{2\sigma_p^2}\right) $$
*   $\sigma_p$: Radius depends on object size.
*   This creates a "soft" target, teaching the network that points *near* the center are also good.

## 3. Corner Pooling (CornerNet)
Standard MaxPool looks for the strongest feature in a local window.
*   **Problem:** A corner (e.g., top-left) often has no visual evidence *at* the corner location (it's empty space). The evidence is to the right (top edge) and below (left edge).
*   **Corner Pooling:**
    *   Max-pool horizontally from right to left.
    *   Max-pool vertically from bottom to top.
    *   Add the two feature maps.
    *   Allows the corner to "see" the edges defining it.

## 4. RepPoints (Representative Points)
**Idea:** Deformable Convolution meets Object Detection.
*   Instead of a box, represent an object by a set of sample points.
*   The network learns to position these points on the object (e.g., on the boundaries/extremities).
*   Bounding box is computed from the min/max of these points.

## Summary
Anchor-free detectors bridge the gap between keypoint estimation and object detection. CenterNet is particularly elegant due to its simplicity and speed (removing NMS).
