# Day 25 Deep Dive: Hungarian Matching & Object Queries

## 1. Object Queries: The "Slots"
What are Object Queries?
*   Learnable embeddings (vectors) initialized randomly.
*   Think of them as **"Question Askers"** or **"Slots"**.
*   Query 1 might learn to ask: "Is there a large object in the top-left?"
*   Query 2 might ask: "Is there a small person in the center?"
*   The Decoder updates these queries by attending to the image features until they contain the answer (Class + Box).

## 2. The Hungarian Algorithm
A combinatorial optimization algorithm to solve the Assignment Problem.
*   **Cost Matrix:** $N \times M$ matrix where entry $(i, j)$ is the cost of assigning prediction $i$ to ground truth $j$.
*   **Goal:** Select assignments such that each row/column is used at most once and total cost is minimized.
*   **Implementation:** `scipy.optimize.linear_sum_assignment`.

## 3. Generalized IoU (GIoU) Loss
**Problem:** Standard IoU is 0 if boxes don't overlap. Gradient is 0.
**Solution:**
$$ GIoU = IoU - \frac{|C \setminus (A \cup B)|}{|C|} $$
*   $C$: Smallest enclosing box covering $A$ and $B$.
*   If no overlap, GIoU moves boxes closer to each other.
*   Crucial for DETR because initial random boxes don't overlap.

## 4. Positional Encodings in DETR
*   **Spatial Positional Encodings** are added to the feature map **at every layer** of the Transformer (unlike NLP where it's only at input).
*   Crucial because Self-Attention is permutation invariant, but detection is highly spatial.

## Summary
DETR turns detection into a direct set prediction problem. The magic lies in the Object Queries (slots) and the Bipartite Matching (unique assignment), which eliminates the need for NMS.
