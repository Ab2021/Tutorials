# Day 25: Detection Transformers (DETR)

## 1. Paradigm Shift
**Traditional Detection:** Hand-crafted components (Anchors, NMS, RPN).
**DETR (DEtection TRansformer):** End-to-End Set Prediction.
*   Treats detection as a direct **set prediction problem**.
*   Removes Anchors and NMS entirely.

## 2. Architecture
1.  **Backbone (CNN):** ResNet-50 extracts features ($H/32 \times W/32 \times 2048$).
2.  **Transformer Encoder:**
    *   Flatten features + Positional Encodings.
    *   Self-Attention captures global context.
3.  **Transformer Decoder:**
    *   Input: **Object Queries** (Learnable embeddings, $N=100$).
    *   Cross-Attention: Queries attend to Encoder output.
    *   Output: $N$ box predictions.
4.  **Prediction Heads:**
    *   FFN predicts Class + Box Coordinates $(x, y, w, h)$.

## 3. Bipartite Matching Loss
**Problem:** The model outputs $N$ predictions. We have $M$ ground truth objects ($M < N$). Which prediction matches which object?
**Solution:** Hungarian Algorithm.
*   Find the one-to-one matching that minimizes the total cost.
    $$ \hat{\sigma} = \arg \min_{\sigma} \sum_{i}^{N} L_{match}(y_i, \hat{y}_{\sigma(i)}) $$
*   **Cost:** Class prob + Box L1 distance + GIoU.
*   Unmatched predictions are trained to predict "No Object" ($\varnothing$).

## 4. Deformable DETR
**Problem:** DETR converges slowly (500 epochs) and fails on small objects (low resolution).
**Solution:**
*   **Deformable Attention:** Each query attends to a small set of key sampling points around a reference point (sparse attention).
*   **Multi-Scale Features:** Uses FPN features.
*   **Result:** Converges 10x faster, SOTA accuracy.

## Summary
DETR simplified the detection pipeline by replacing heuristics (anchors/NMS) with learnable components (Transformers/Queries). It is the foundation of modern detection research.
