# Day 37 Deep Dive: Instant-NGP & Gaussian Splatting

## 1. The Speed Problem
Original NeRF takes **hours/days** to train for a single scene.
*   Reason: The MLP is queried millions of times (Resolution $\times$ Rays per pixel $\times$ Samples per ray).

## 2. Instant-NGP (Neural Graphics Primitives)
**Idea:** Replace the deep MLP with a **Multi-Resolution Hash Grid**.
*   **Hash Encoding:**
    *   Store trainable feature vectors in a hash table at multiple resolution levels.
    *   Map coordinate $x$ to hash index $h(x)$. Look up feature vector.
    *   Interpolate features.
*   **Tiny MLP:** Use a very small MLP (2 layers) to decode the features.
*   **Result:** Trains in **seconds** (5s to look good, 1 min for high quality).

## 3. 3D Gaussian Splatting (2023)
**Idea:** Move away from MLPs entirely. Back to explicit representation.
*   Represent scene as millions of **3D Gaussians**.
    *   Parameters: Position $\mu$, Covariance $\Sigma$, Color $c$, Opacity $\alpha$.
*   **Rasterization:** Project Gaussians to 2D screen (Splatting).
*   **Training:** Optimize parameters of all Gaussians via Gradient Descent to match images.
*   **Result:** Real-time rendering (100+ FPS) with NeRF quality.

## 4. NeRF in the Wild (NeRF-W)
**Problem:** Real photos have variable lighting (time of day) and transient objects (pedestrians).
**Solution:**
*   **Appearance Embedding:** Learn a vector for each image to handle lighting changes.
*   **Transient Head:** A separate output to predict "uncertainty" or transient density (ignore pedestrians).

## Summary
The field moved from "Slow but Accurate" (NeRF) to "Fast and Accurate" (Instant-NGP, Gaussian Splatting), enabling real-time 3D capture on mobile devices.
