# Day 5 Interview Questions: Segmentation Basics

## Q1: Explain Otsu's method. What assumption does it make?
**Answer:**
Otsu's method automatically finds the optimal global threshold $T$ by maximizing the **inter-class variance** (separation between foreground and background) or minimizing the **intra-class variance**.
*   **Assumption:** The image histogram is **bimodal** (two distinct peaks).
*   It fails if the histogram is unimodal or if the object/background sizes are extremely unbalanced.

## Q2: What is the difference between semantic segmentation and instance segmentation?
**Answer:**
*   **Semantic Segmentation:** Classifies each pixel into a category (e.g., "car", "road"). All cars have the same label/color.
*   **Instance Segmentation:** Classifies each pixel *and* distinguishes between different instances of the same category. Car #1 is different from Car #2.

## Q3: How does the Watershed algorithm work? What is its main problem?
**Answer:**
*   **Concept:** Treats image intensity as a topographic surface. Light pixels are peaks, dark pixels are valleys.
*   **Process:** Floods the surface from local minima (or markers). Where waters from different basins meet, a dam (boundary) is built.
*   **Main Problem:** **Over-segmentation** due to noise or local texture irregularities.
*   **Solution:** Use **Marker-controlled Watershed** where seeds are manually or automatically placed on sure foreground/background.

## Q4: How does K-Means segmentation differ from Region Growing?
**Answer:**
*   **K-Means:** Based on **color similarity** in the feature space. It does not enforce spatial connectivity (pixels in different corners can belong to the same cluster).
*   **Region Growing:** Based on **spatial connectivity**. It starts from a seed and expands to neighbors. It guarantees connected regions.

## Q5: What is GrabCut?
**Answer:**
An interactive segmentation technique based on **Graph Cuts**.
1.  User initializes with a bounding box.
2.  Algorithm models foreground/background using Gaussian Mixture Models (GMMs).
3.  It builds a graph where pixels are nodes and edges represent similarity.
4.  It finds the **Min-Cut** to separate foreground from background.
5.  Iterates to refine.

## Q6: Why use HSV color space for segmentation instead of RGB?
**Answer:**
*   **RGB:** Color and intensity are correlated. Shadows or lighting changes affect R, G, and B values significantly.
*   **HSV:** Separates **Color (Hue)** from **Intensity (Value)**.
*   **Benefit:** You can segment an object based on its Hue (e.g., "red") regardless of whether it's in shadow or bright light, making it more robust to lighting conditions.

## Q7: What are Superpixels?
**Answer:**
Groups of connected pixels that share similar colors or textures.
*   They replace the rigid pixel grid with a perceptual grid.
*   **Benefit:** Reduces the number of primitives for subsequent algorithms (e.g., from 1 million pixels to 1000 superpixels), speeding up processing significantly.

## Q8: Implement a simple region growing algorithm.
**Answer:**
```python
def region_growing(img, seed, threshold):
    h, w = img.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    output = np.zeros((h, w), dtype=np.uint8)
    
    stack = [seed]
    visited[seed] = 1
    seed_val = img[seed]
    
    while stack:
        x, y = stack.pop()
        output[x, y] = 255
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                if abs(int(img[nx, ny]) - int(seed_val)) < threshold:
                    visited[nx, ny] = 1
                    stack.append((nx, ny))
                    
    return output
```

## Q9: What is the "Mean Shift" algorithm?
**Answer:**
A non-parametric clustering algorithm.
*   It places a window (kernel) on the data points.
*   It computes the **mean** of the points within the window.
*   It **shifts** the window center to the mean.
*   Repeats until convergence (peak of density).
*   Used for segmentation (clustering pixels) and tracking.

## Q10: How do you handle noise before thresholding?
**Answer:**
Apply **Gaussian Blur** or **Median Blur**.
*   Noise creates local intensity spikes that can be misclassified by a threshold.
*   Smoothing the image creates a cleaner histogram, making it easier to find a good separation point (e.g., for Otsu's method).
