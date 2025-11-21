# Day 37: Neural Radiance Fields (NeRF)

## 1. Implicit Neural Representations
**Idea:** Instead of storing a 3D model as a mesh or point cloud, store it as a **Function** (Neural Network).
*   **Input:** Coordinate $(x, y, z)$ and Viewing Direction $(\theta, \phi)$.
*   **Output:** Color $(r, g, b)$ and Density $\sigma$.
*   $F_\Theta: (x, y, z, \theta, \phi) \to (r, g, b, \sigma)$.

## 2. Volume Rendering
How to generate an image from this function?
*   **Ray Marching:** Shoot a ray $r(t) = o + td$ from the camera through each pixel.
*   Sample points along the ray.
*   Query the network for color $c_i$ and density $\sigma_i$ at each point.
*   **Accumulate:**
    $$ C(r) = \sum_{i=1}^N T_i (1 - \exp(-\sigma_i \delta_i)) c_i $$
    *   $T_i$: Transmittance (probability that ray reaches point $i$ without hitting anything).
    *   $(1 - \exp(-\sigma_i \delta_i))$: Opacity at point $i$.

## 3. Positional Encoding
**Problem:** Neural networks are biased towards low frequencies (Spectral Bias). They produce blurry 3D shapes.
**Solution:** Map inputs $(x, y, z)$ to higher dimensions using high-frequency sinusoids.
$$ \gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)) $$
*   This allows the MLP to learn high-frequency details (sharp edges/textures).

## 4. Training
*   **Data:** A set of images of a scene with known camera poses (COLMAP).
*   **Loss:** MSE between rendered pixel color and ground truth pixel color.
    $$ L = \sum_{r \in R} || C(r) - C_{GT}(r) ||^2 $$
*   **Result:** The network "memorizes" the 3D scene.

```python
# Conceptual NeRF Forward Pass
def render_ray(model, ray_origin, ray_dir):
    t_vals = torch.linspace(near, far, N_samples)
    points = ray_origin + ray_dir * t_vals
    
    # Positional Encoding
    points_enc = positional_encoding(points)
    
    # Query MLP
    rgb, sigma = model(points_enc)
    
    # Volume Rendering Equation
    # ... (Accumulate colors based on density) ...
    return pixel_color
```

## Summary
NeRF revolutionized 3D reconstruction by showing that a simple MLP can represent complex scenes with photorealistic quality, provided we use Volume Rendering and Positional Encoding.
