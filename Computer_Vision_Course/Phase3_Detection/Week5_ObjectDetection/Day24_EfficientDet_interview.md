# Day 24 Interview Questions: EfficientDet

## Q1: What is the main contribution of EfficientDet?
**Answer:**
1.  **BiFPN:** A bidirectional feature pyramid network with learnable weights for feature fusion.
2.  **Compound Scaling:** A principled method to scale up the backbone, feature network, and box/class prediction networks simultaneously.

## Q2: How does BiFPN differ from standard FPN?
**Answer:**
*   **FPN:** One-way information flow (Top-Down).
*   **BiFPN:** Two-way flow (Top-Down and Bottom-Up).
*   **Fusion:** FPN sums features. BiFPN uses **Weighted Feature Fusion** (learnable weights).
*   **Structure:** BiFPN removes nodes with single inputs and adds skip connections.

## Q3: Why use "Fast Normalized Fusion" instead of Softmax?
**Answer:**
*   Softmax involves exponentials ($e^x$), which are computationally expensive on hardware accelerators.
*   Fast Normalized Fusion ($\frac{w}{\sum w}$) uses simple addition and division, achieving similar accuracy with significantly higher speed (up to 30% faster).

## Q4: Explain the "Compound Scaling" for detection.
**Answer:**
It uses a single coefficient $\phi$ to determine:
*   Backbone network (EfficientNet-B$\phi$).
*   BiFPN depth ($D_{bifpn} = 3 + \phi$) and width ($W_{bifpn} = 64 \cdot (1.35^\phi)$).
*   Input resolution ($R = 512 + \phi \cdot 128$).
*   This ensures all parts of the network grow in balance.

## Q5: Why is EfficientDet efficient in FLOPs but sometimes slower in latency than YOLO?
**Answer:**
*   EfficientDet relies heavily on **Depthwise Separable Convolutions**.
*   While DS-Convs have very low FLOPs, they have low **Arithmetic Intensity** (ratio of compute to memory access).
*   On GPUs, memory bandwidth is often the bottleneck. Standard convolutions (used in YOLO) are more compute-heavy but utilize the GPU cores better.

## Q6: What is the role of the Backbone in a detector?
**Answer:**
To extract a hierarchy of features from the image.
*   Low-level features (edges) from early layers.
*   High-level features (semantics) from deep layers.
*   EfficientDet uses EfficientNet, which is better than ResNet at extracting features with fewer parameters.

## Q7: How does Weighted Feature Fusion help?
**Answer:**
*   Features from different resolutions contribute unequally to the output.
*   Example: For a large object, the low-res (semantic) features are more important. For a small object, high-res features are key.
*   Learnable weights allow the network to decide dynamically which features to trust.

## Q8: What is PANet?
**Answer:**
Path Aggregation Network.
*   An improvement over FPN.
*   Adds a bottom-up path *after* the top-down FPN path.
*   Shortens the information path between low-level features and top-level features.
*   BiFPN simplifies and optimizes PANet.

## Q9: Why do we need to scale the Box/Class network depth?
**Answer:**
*   As the backbone and BiFPN get larger/deeper, the features become more complex.
*   The final prediction heads need more capacity (layers) to process these complex features accurately.

## Q10: Implement Fast Normalized Fusion.
**Answer:**
```python
def fast_normalized_fusion(inputs, weights):
    # inputs: list of tensors [I1, I2]
    # weights: list of learnable scalars [w1, w2]
    
    # Ensure weights >= 0
    w_relu = [F.relu(w) for w in weights]
    
    # Normalize
    w_sum = sum(w_relu) + 0.0001
    
    # Weighted Sum
    output = sum([w * I for w, I in zip(w_relu, inputs)]) / w_sum
    return output
```
