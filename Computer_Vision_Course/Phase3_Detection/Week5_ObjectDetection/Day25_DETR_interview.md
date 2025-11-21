# Day 25 Interview Questions: DETR

## Q1: How does DETR eliminate the need for NMS?
**Answer:**
*   Traditional detectors output thousands of redundant boxes (many anchors per object). NMS cleans this up.
*   DETR uses **Bipartite Matching** during training.
*   This forces the model to predict **exactly one** box per ground truth object.
*   The "Object Queries" learn to communicate via self-attention to ensure they don't predict the same object twice (duplicate removal is learned).

## Q2: What are "Object Queries"?
**Answer:**
*   A fixed set of learnable embeddings (e.g., 100) fed into the Decoder.
*   They act as "slots" that the model fills with detected objects.
*   Each query specializes in detecting objects in certain regions or sizes.

## Q3: Why does DETR converge slowly?
**Answer:**
*   **Attention Map Initialization:** The attention weights start almost uniform. It takes a long time for the queries to learn to focus on specific sparse regions of the image.
*   **Small Objects:** Standard DETR uses a low-resolution feature map ($H/32$), making small objects hard to see.
*   *Deformable DETR solves this by initializing attention near reference points.*

## Q4: Explain the Bipartite Matching Loss.
**Answer:**
*   It is a loss calculated between the set of $N$ predictions and the set of $M$ ground truth objects.
*   First, the Hungarian Algorithm finds the optimal 1-to-1 matching that minimizes the cost.
*   Then, standard loss (Cross-Entropy + Box Regression) is applied only to the matched pairs.
*   Unmatched predictions are penalized to predict "No Object" (background).

## Q5: What is the difference between Self-Attention and Cross-Attention in DETR?
**Answer:**
*   **Self-Attention (Encoder):** Pixels attend to other pixels. Captures global context and object relationships.
*   **Self-Attention (Decoder):** Queries attend to other queries. Helps avoid duplicates.
*   **Cross-Attention (Decoder):** Queries attend to Encoder output (Image features). This is where the query extracts information about the object.

## Q6: Why is GIoU used instead of L1 loss for box regression?
**Answer:**
*   L1 loss depends on the scale of the box (large boxes have large errors).
*   IoU is scale-invariant.
*   However, IoU is 0 for non-overlapping boxes (no gradient).
*   GIoU fixes this by providing a gradient even when boxes are far apart, pulling them together.

## Q7: Can DETR detect more than 100 objects?
**Answer:**
**No.**
*   The number of object queries $N$ (e.g., 100) is a hard limit fixed at architecture design.
*   If an image has 101 objects, the model will miss at least one.
*   However, 100 is usually sufficient for COCO (avg 7 objects). For dense crowds, $N$ must be increased.

## Q8: What is the role of the CNN backbone in DETR?
**Answer:**
*   Transformers are computationally expensive on raw pixels ($O(H^2 W^2)$).
*   The CNN (ResNet) reduces the image from $1024 \times 1024$ to a manageable feature map ($32 \times 32$).
*   DETR operates on these high-level features.

## Q9: How does Deformable DETR improve speed?
**Answer:**
*   Standard Attention looks at **all** pixels.
*   Deformable Attention only looks at a small set of **sampling points** (e.g., 4) around a reference point.
*   This reduces complexity from quadratic to linear, allowing the use of multi-scale feature maps (high resolution) for better small object detection.

## Q10: Implement the Hungarian Matcher call.
**Answer:**
```python
from scipy.optimize import linear_sum_assignment

def match(cost_matrix):
    # cost_matrix: (N_pred, M_gt)
    # returns: (row_ind, col_ind)
    # row_ind: indices of predictions
    # col_ind: indices of matched GT
    return linear_sum_assignment(cost_matrix)
```
