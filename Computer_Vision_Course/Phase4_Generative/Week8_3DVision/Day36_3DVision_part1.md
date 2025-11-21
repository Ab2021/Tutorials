# Day 36 Deep Dive: PointNet++ & Graph CNNs

## 1. PointNet++ (2017)
**Limitation of PointNet:** It treats all points globally. It misses **local structure** (like CNNs capture local patterns).
**Solution:** Hierarchical PointNet.
1.  **Sampling:** Select $N'$ centroids (Farthest Point Sampling).
2.  **Grouping:** Find $K$ nearest neighbors for each centroid (Ball Query).
3.  **PointNet:** Apply a mini-PointNet to each local group to extract features.
4.  **Repeat:** Stack these layers to learn hierarchical features.

## 2. Graph Convolutional Networks (GCN) for 3D
**Idea:** Treat the point cloud (or mesh) as a Graph $G=(V, E)$.
*   **Dynamic Graph CNN (DGCNN):**
    *   Construct a k-NN graph dynamically in feature space.
    *   **EdgeConv:** Convolve over edges. $h_i' = \max_{j \in N(i)} \text{MLP}(h_i, h_j - h_i)$.
    *   Captures local geometric structure better than PointNet.

## 3. 3D Data Augmentation
Crucial for 3D tasks.
*   **Rotation:** Rotate around Up-axis (Gravity).
*   **Jitter:** Add Gaussian noise to coordinates.
*   **Scaling:** Randomly scale the object.
*   **Shuffling:** Randomly permute the order of points (Network must be invariant!).

## 4. VoxelNet & 3D CNNs
*   Convert point cloud to Voxels.
*   Apply 3D Convolutions ($3 \times 3 \times 3$ kernels).
*   **Pros:** Captures spatial structure perfectly.
*   **Cons:** Memory hungry. Most voxels are empty (Sparse).
*   **Sparse ConvNets:** Only compute at non-empty locations.

## Summary
While PointNet is the baseline, modern architectures (PointNet++, DGCNN, SparseConv) focus on capturing local neighborhoods and hierarchical structures, similar to how 2D CNNs work.
