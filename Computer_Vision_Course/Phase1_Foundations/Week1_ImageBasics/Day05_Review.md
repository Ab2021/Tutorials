# Day 5: Image Segmentation Basics

## 1. Introduction to Segmentation
**Goal:** Partition an image into multiple segments (sets of pixels) to simplify representation and make it more meaningful.
*   **Input:** Image.
*   **Output:** Mask where each pixel has a label.

## 2. Thresholding
Simplest method. Separate objects from background based on intensity.

### Global Thresholding
$$ g(x, y) = \begin{cases} 1 & \text{if } f(x, y) > T \\ 0 & \text{otherwise} \end{cases} $$

### Otsu's Method (Automated Global)
Finds optimal threshold $T$ that minimizes intra-class variance (or maximizes inter-class variance).
*   Assumes bimodal histogram (foreground + background).

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)

# Global Thresholding (T=127)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's Thresholding
ret2, thresh2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Adaptive Thresholding
Threshold $T$ is calculated for smaller regions. Good for uneven lighting.
*   **Mean:** $T$ = mean of neighborhood - C.
*   **Gaussian:** $T$ = weighted sum of neighborhood - C.

```python
thresh3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
```

## 3. Region-Based Segmentation

### Region Growing
Start with seed points. Add neighbors if they are similar (intensity difference < threshold).
*   **Pros:** Connected regions.
*   **Cons:** Sensitive to noise and seed selection.

### Watershed Algorithm
Treat image as a topographic map (brightness = height).
1.  Find markers (sure foreground, sure background).
2.  Flood from markers.
3.  Barriers ("dams") built where waters meet = segmentation lines.
*   Great for separating touching objects (e.g., coins).

```python
# Watershed Steps
# 1. Binarize (Otsu)
# 2. Distance Transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# 3. Threshold distance map to get sure foreground
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# 4. Subtract sure_fg from background to get unknown region
unknown = cv2.subtract(sure_bg, sure_fg)
# 5. Label markers
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
# 6. Apply Watershed
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0] # Mark boundaries
```

## 4. Clustering-Based Segmentation

### K-Means Clustering
Cluster pixels based on color $(R, G, B)$ or color+position $(R, G, B, x, y)$.
1.  Initialize $K$ centroids.
2.  Assign pixels to nearest centroid.
3.  Update centroids.
4.  Repeat.

```python
# Flatten image
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Reconstruct
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
```

### Mean Shift Segmentation
Non-parametric clustering.
*   Find modes of the probability density function.
*   Shift window towards the mean of data points within it.
*   **Pros:** No need to specify $K$. Edge-preserving.
*   **Cons:** Slower.

## 5. Graph-Based Segmentation
Represent image as a graph $G=(V, E)$.
*   **Nodes:** Pixels.
*   **Edges:** Similarity between pixels.
*   **Goal:** Cut graph into subgraphs such that cut cost is minimized (Normalized Cuts).

## Summary
Classical segmentation relies on intensity (thresholding), connectivity (region growing), or color clustering (K-Means). These are foundational for understanding modern deep learning segmentation (U-Net, Mask R-CNN).
