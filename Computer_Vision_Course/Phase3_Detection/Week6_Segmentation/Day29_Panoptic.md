# Day 29: Panoptic Segmentation

## 1. Problem Definition
**Goal:** Unify Semantic and Instance Segmentation.
*   **Things:** Countable objects (Car, Person). Need Instance IDs.
*   **Stuff:** Amorphous regions (Sky, Road, Grass). Need Class Labels only.
*   **Output:** Each pixel has a tuple $(l, z)$ where $l$ is class label and $z$ is instance ID.

## 2. Panoptic FPN (2019)
**Idea:** Add a Semantic Segmentation branch to Mask R-CNN.
*   **Backbone:** ResNet-FPN.
*   **Instance Branch:** Mask R-CNN (RPN + RoIAlign + Heads). Handles "Things".
*   **Semantic Branch:**
    *   Takes features from all FPN levels (P2-P5).
    *   Upsamples them to 1/4 scale.
    *   Sums them up.
    *   Predicts semantic logits. Handles "Stuff".
*   **Fusion:** Combine Instance and Semantic outputs.
    *   Resolve overlaps (Instance mask usually overrides Semantic mask).

## 3. DETR for Panoptic
**Idea:** Use Transformer queries to predict masks directly.
*   **Box Head:** Predicts boxes for "Things".
*   **Mask Head:** Attention maps from the Multi-Head Attention are used to generate binary masks for each query.
*   **Unified:** No distinction between Things and Stuff. "Stuff" classes are just queries that predict a box covering the whole region (or no box).

## 4. Panoptic Quality (PQ) Metric
$$ PQ = \frac{\sum_{(p, g) \in TP} IoU(p, g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|} $$
*   Combines **Recognition Quality** (F1 score) and **Segmentation Quality** (Avg IoU).
*   Strict metric: A match is only counted if IoU > 0.5.

## Summary
Panoptic Segmentation provides the most complete understanding of a scene. It is crucial for autonomous driving (drivable area + obstacles).
