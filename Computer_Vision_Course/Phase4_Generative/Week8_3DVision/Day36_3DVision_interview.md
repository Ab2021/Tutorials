# Day 36 Interview Questions: 3D Vision

## Q1: Why is "Permutation Invariance" important for Point Clouds?
**Answer:**
*   A point cloud is a **Set** of points. The order doesn't matter.
*   Input: $[P_1, P_2, P_3]$ is the same object as $[P_3, P_1, P_2]$.
*   Standard MLPs or RNNs depend on order.
*   PointNet uses a symmetric function (Max Pooling) which yields the same result regardless of input order.

## Q2: What is the difference between Point Cloud and Voxel representations?
**Answer:**
*   **Point Cloud:** Continuous coordinates. Sparse. Efficient for surfaces. Hard to process (unordered).
*   **Voxel:** Discretized grid. Dense. Inefficient (mostly empty space). Easy to process (3D CNN).

## Q3: Explain Chamfer Distance.
**Answer:**
*   A metric to compare two point clouds.
*   It sums the squared distances from each point in Set A to its nearest neighbor in Set B, and vice versa.
*   It is differentiable, making it suitable as a loss function for 3D reconstruction.

## Q4: How does PointNet++ improve on PointNet?
**Answer:**
*   PointNet processes all points globally (one max pool). It fails to capture local features (e.g., the shape of a car handle).
*   PointNet++ applies PointNet recursively on local neighborhoods (grouping points), allowing it to learn hierarchical features like a CNN.

## Q5: What is "Farthest Point Sampling" (FPS)?
**Answer:**
*   A sampling strategy to select a subset of points that covers the shape well.
*   Start with a random point.
*   Iteratively select the point that is farthest from the already selected points.
*   Ensures uniform coverage (unlike random sampling which might cluster).

## Q6: What is the T-Net in PointNet?
**Answer:**
*   A mini-network that predicts a $3 \times 3$ (or $64 \times 64$) transformation matrix.
*   It multiplies the input points/features by this matrix to align them to a canonical orientation.
*   Makes the network invariant to rotation/viewpoint.

## Q7: Why are 3D meshes hard for Deep Learning?
**Answer:**
*   Meshes are graphs with varying topology (different number of neighbors per vertex).
*   Standard convolution requires a fixed grid structure.
*   Requires specialized Graph Neural Networks (GNNs) or converting to other formats.

## Q8: What is "Earth Mover's Distance" (EMD) for Point Clouds?
**Answer:**
*   Another metric for comparing point clouds.
*   Solves the assignment problem (transportation problem) to match points in A to points in B 1-to-1.
*   More sensitive to density than Chamfer Distance but computationally expensive ($O(N^2)$ or $O(N^3)$).

## Q9: How do RGB-D cameras work?
**Answer:**
*   They capture Color (RGB) and Depth (D).
*   **Structured Light (Kinect v1):** Projects a known pattern and measures distortion.
*   **Time of Flight (Kinect v2):** Measures time taken for light pulse to return.
*   **Stereo:** Uses two cameras and disparity.

## Q10: Implement Chamfer Distance (Naive).
**Answer:**
```python
def chamfer_distance(p1, p2):
    # p1: (N, 3), p2: (M, 3)
    # Expand dims to create distance matrix (N, M)
    dist = torch.cdist(p1, p2) # Euclidean distance
    
    min_dist_p1, _ = torch.min(dist, dim=1) # Nearest in p2 for each p1
    min_dist_p2, _ = torch.min(dist, dim=0) # Nearest in p1 for each p2
    
    return torch.mean(min_dist_p1) + torch.mean(min_dist_p2)
```
