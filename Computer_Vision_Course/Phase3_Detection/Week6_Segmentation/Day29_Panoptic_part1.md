# Day 29 Deep Dive: Panoptic Architectures

## 1. Panoptic-DeepLab
**Bottom-Up Approach.**
*   **Semantic Head:** Predicts class label for every pixel.
*   **Instance Head:** Predicts:
    1.  **Center Heatmap:** Center of object.
    2.  **Offset Map:** Vector from each pixel to its instance center.
*   **Grouping:** Pixels are grouped to the nearest center based on predicted offsets.
*   **Benefit:** Simple, fast, anchor-free.

## 2. MaskFormer (2021)
**Idea:** Segmentation is a mask classification problem.
*   **Pixel-level module:** Extracts pixel embeddings (CNN/Transformer).
*   **Transformer Decoder:** Generates $N$ mask embeddings (Queries).
*   **Prediction:** Dot product between pixel embeddings and mask embeddings $\to$ $N$ binary masks.
*   **Classification:** Each mask is classified into Class $C$ or "No Object".
*   **Result:** Unifies Semantic and Instance segmentation elegantly. SOTA performance.

## 3. UPSNet (Unified Panoptic Segmentation)
**Idea:** Explicitly model the interaction between Things and Stuff.
*   **Heads:** Semantic Head + Instance Head.
*   **Panoptic Head:** Takes logits from both.
    *   Adds a "Unknown" class for void regions.
    *   Resolves conflicts (e.g., if a pixel is both "Car" and "Road", "Car" wins).

## 4. Kirillov's Fusion Heuristic
How to merge Mask R-CNN (Instance) and FCN (Semantic) outputs?
1.  **Paste Instances:** Place high-confidence instance masks on a blank canvas.
2.  **Resolve Overlaps:** If two instances overlap, sort by confidence or depth.
3.  **Fill Gaps:** Fill remaining empty pixels with the Semantic Segmentation output.
4.  **Filter:** Remove small "stuff" regions that are covered by instances.

## Summary
The field is moving from heuristic fusion (Panoptic FPN) to fully unified architectures (MaskFormer) where the distinction between Things and Stuff is minimized.
