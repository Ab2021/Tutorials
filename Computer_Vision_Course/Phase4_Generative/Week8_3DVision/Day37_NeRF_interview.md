# Day 37 Interview Questions: NeRF

## Q1: What is the difference between Explicit and Implicit 3D representations?
**Answer:**
*   **Explicit:** Geometry is stored directly (Mesh vertices, Voxel grids, Point clouds). Memory scales with resolution.
*   **Implicit:** Geometry is defined as a function (e.g., $f(x,y,z) = 0$ for surface). NeRF is implicit. Memory scales with network complexity, not resolution.

## Q2: Why does NeRF need Positional Encoding?
**Answer:**
*   Standard MLPs are "Universal Function Approximators", but in practice, they struggle to learn high-frequency functions (sharp edges) from low-dimensional inputs $(x, y, z)$.
*   This is known as **Spectral Bias**.
*   Positional Encoding maps the input to a higher-dimensional space using sine/cosine functions of different frequencies, enabling the network to capture fine details.

## Q3: Explain Volume Rendering in simple terms.
**Answer:**
*   It's like looking through a fog.
*   We shoot a ray from the eye.
*   As the ray travels, it hits particles.
*   Each particle has a color and a density (how much light it blocks).
*   We sum up the color contributions of all particles along the ray, weighted by how much light reached them (Transmittance).

## Q4: What is "View Dependence" in NeRF?
**Answer:**
*   The color of a point depends on the viewing angle (e.g., specular reflections on a shiny car).
*   NeRF inputs the viewing direction $(\theta, \phi)$ along with position.
*   This allows it to render realistic lighting effects like reflections, which standard photogrammetry (meshes) struggles with.

## Q5: Why is Gaussian Splatting faster than NeRF?
**Answer:**
*   **NeRF:** Requires querying an MLP hundreds of times *per pixel* (Ray Marching). Expensive.
*   **Gaussian Splatting:** Projects 3D Gaussians to 2D (Rasterization). This is a sorting operation, which is highly optimized on GPUs (similar to standard triangle rasterization). No MLP queries during rendering.

## Q6: What is the input to a NeRF training pipeline?
**Answer:**
*   A set of images.
*   Camera poses (Extrinsics) and intrinsics for each image.
*   Usually obtained using Structure-from-Motion (SfM) software like COLMAP.

## Q7: How does Instant-NGP achieve such high speed?
**Answer:**
*   It replaces the large, deep MLP with a **Hash Grid**.
*   The Hash Grid acts as a learnable memory that stores spatial features.
*   The MLP becomes tiny (shallow), serving only to decode these features.
*   Memory access is faster than matrix multiplication.

## Q8: Can NeRF handle dynamic scenes (video)?
**Answer:**
*   Standard NeRF assumes a static scene.
*   **Dynamic NeRF (D-NeRF):** Adds a time dimension $t$.
*   Often uses a "Deformation Network" that maps points at time $t$ to a canonical space at time $0$.

## Q9: What is "Hierarchical Sampling" in NeRF?
**Answer:**
*   Sampling points uniformly along a ray is wasteful (most of the ray is empty space).
*   **Coarse Network:** Samples uniformly. Finds where the density is high.
*   **Fine Network:** Samples more points in the high-density regions.
*   Improves efficiency and quality.

## Q10: Implement the Positional Encoding function.
**Answer:**
```python
def positional_encoding(x, L=10):
    # x: (N, 3)
    out = [x]
    for i in range(L):
        freq = 2.0 ** i
        out.append(torch.sin(freq * np.pi * x))
        out.append(torch.cos(freq * np.pi * x))
    return torch.cat(out, dim=-1)
```
