# Day 5 Deep Dive: Advanced Classical Segmentation

## 1. Graph Cut (Min-Cut/Max-Flow)
**Concept:**
*   **Source (S):** Object terminal.
*   **Sink (T):** Background terminal.
*   **t-links:** Connect pixels to S or T (based on likelihood).
*   **n-links:** Connect neighboring pixels (penalty for discontinuity).
*   **Min-Cut:** Find cut with minimum cost that separates S from T.

**GrabCut:**
Iterative Graph Cut.
1.  User draws box around object.
2.  GMM models foreground/background color distribution.
3.  Graph Cut estimates segmentation.
4.  Update GMMs based on new segmentation.
5.  Repeat.

```python
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]
```

## 2. Superpixels (SLIC)
**Simple Linear Iterative Clustering (SLIC).**
*   Group pixels into perceptually meaningful atomic regions.
*   Replaces rigid pixel grid with irregular superpixel grid.
*   Uses K-Means in 5D space $(L, a, b, x, y)$.
*   **Distance:**
    $$ D = \sqrt{d_c^2 + \left(\frac{d_s}{S}\right)^2 m^2} $$
    *   $d_c$: Color distance.
    *   $d_s$: Spatial distance.
    *   $m$: Compactness factor.

**Benefits:**
*   Reduces complexity for downstream tasks (graph cuts, object detection).
*   Adheres to image boundaries.

## 3. Felzenszwalb's Efficient Graph-Based Segmentation
*   Bottom-up merging of regions.
*   Predicate for merging: Is the difference *between* two components smaller than the difference *within* the components?
*   Fast and captures non-local properties.

## 4. Comparison of Methods

| Method | Principle | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Otsu** | Histogram | Fast, Automatic | Fails on noisy/uneven light |
| **Watershed** | Morphology | Separates touching objects | Over-segmentation |
| **K-Means** | Clustering | Simple color grouping | Needs K, no spatial coherence |
| **Mean Shift** | Density Est. | Good boundaries, no K | Slow |
| **GrabCut** | Graph Cut | Interactive, High quality | Slow, needs user input |

## 5. Evaluation Metrics (Classical)
*   **IoU (Intersection over Union):** $ \frac{A \cap B}{A \cup B} $
*   **Dice Coefficient:** $ \frac{2 |A \cap B|}{|A| + |B|} $
*   **Pixel Accuracy:** Correct pixels / Total pixels.
