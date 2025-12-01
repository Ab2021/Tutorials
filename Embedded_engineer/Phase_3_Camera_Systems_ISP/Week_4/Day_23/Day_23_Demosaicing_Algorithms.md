# Day 23: Demosaicing & Interpolation Algorithms
## Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development

---

## 🎯 Learning Objectives
1. **Understand** the Bayer Color Filter Array (CFA) and the need for demosaicing
2. **Implement** basic interpolation algorithms (Nearest Neighbor, Bilinear)
3. **Develop** advanced edge-aware algorithms (Malvar-He-Cutler, AHD)
4. **Analyze** demosaicing artifacts (Zipper effect, False Color, Moiré)
5. **Optimize** interpolation performance using SIMD/Vectorization
6. **Evaluate** algorithm quality using PSNR/SSIM metrics

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Raw Camera Capture
*   **Software:** C/C++ Compiler, OpenCV (for ground truth comparison)
*   **Knowledge:** Convolution, Gradients, Nyquist-Shannon Sampling Theorem

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Bayer CFA
Most sensors use a Bayer pattern (RGGB, BGGR, GBRG, GRBG).
*   **Green:** 50% of pixels (Human eye is most sensitive to green/luminance).
*   **Red/Blue:** 25% each.
*   **Problem:** At each pixel location, we only measure ONE color. We need to estimate the other two.

### 🔹 Part 2: Demosaicing Algorithms

#### 2.1 Non-Adaptive (Linear)
*   **Nearest Neighbor:** Copy value from neighbor. Fast, but blocky.
*   **Bilinear Interpolation:** Average of 2 or 4 neighbors. Smooths edges, causes "zipper" artifacts.

#### 2.2 Adaptive (Non-Linear)
*   **Edge-Directed:** Calculate gradients (Horizontal vs Vertical). Interpolate *along* the edge, not across it.
*   **Constant Hue Assumption:** In a local area, the ratio R/G and B/G is constant. We interpolate the *difference* (G-R) or (G-B) rather than the raw values.
*   **Malvar-He-Cutler (High Quality Linear):** Adds a Laplacian correction term (2nd derivative) to Bilinear.
*   **AHD (Adaptive Homogeneity-Directed):** Selects the direction with the most "homogeneity" (smoothness) in the CIELAB color space.

### 🔹 Part 3: Artifacts
*   **Zipper Effect:** On/Off pattern at edges (like a zipper). Caused by averaging across an edge.
*   **False Color:** High-frequency patterns (stripes) interpreted as color.
*   **Moiré:** Aliasing patterns when scene frequency > Nyquist frequency.

---

## 💻 Implementation Examples

### Example 1: Bilinear Interpolation (Reference)

```cpp
/**
 * @brief Bilinear Demosaicing (RGGB)
 */
void demosaic_bilinear(const RawImage& raw, RGBImage& rgb) {
    int w = raw.width;
    int h = raw.height;
    
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            uint16_t val = raw.data[idx];
            
            float r, g, b;
            
            // Determine pixel type (RGGB)
            if (y % 2 == 0 && x % 2 == 0) { // Red Pixel
                r = val;
                g = (raw.data[idx-1] + raw.data[idx+1] + 
                     raw.data[idx-w] + raw.data[idx+w]) / 4.0f;
                b = (raw.data[idx-w-1] + raw.data[idx-w+1] + 
                     raw.data[idx+w-1] + raw.data[idx+w+1]) / 4.0f;
            }
            else if (y % 2 == 0 && x % 2 == 1) { // Green (Red row)
                r = (raw.data[idx-1] + raw.data[idx+1]) / 2.0f;
                g = val;
                b = (raw.data[idx-w] + raw.data[idx+w]) / 2.0f;
            }
            else if (y % 2 == 1 && x % 2 == 0) { // Green (Blue row)
                r = (raw.data[idx-w] + raw.data[idx+w]) / 2.0f;
                g = val;
                b = (raw.data[idx-1] + raw.data[idx+1]) / 2.0f;
            }
            else { // Blue Pixel
                r = (raw.data[idx-w-1] + raw.data[idx-w+1] + 
                     raw.data[idx+w-1] + raw.data[idx+w+1]) / 4.0f;
                g = (raw.data[idx-1] + raw.data[idx+1] + 
                     raw.data[idx-w] + raw.data[idx+w]) / 4.0f;
                b = val;
            }
            
            rgb.data[idx].r = (uint16_t)r;
            rgb.data[idx].g = (uint16_t)g;
            rgb.data[idx].b = (uint16_t)b;
        }
    }
}
```

### Example 2: Malvar-He-Cutler (High Quality Linear)

This algorithm improves Green interpolation by using the Laplacian of the Red/Blue channel to correct it.
`G_at_R = (G_north + G_south + G_east + G_west)/4 + alpha * (R_center - (R_north + R_south + R_east + R_west)/4)`
Ideally, `alpha = 0.5`.

```cpp
/**
 * @brief Malvar-He-Cutler Demosaicing (Green Channel Only)
 */
void demosaic_mhc_green(const RawImage& raw, float* green_channel) {
    int w = raw.width;
    
    #pragma omp parallel for
    for (int y = 2; y < raw.height - 2; y++) {
        for (int x = 2; x < raw.width - 2; x++) {
            int idx = y * w + x;
            
            // If already Green, just copy
            if ((y + x) % 2 == 1) {
                green_channel[idx] = raw.data[idx];
                continue;
            }
            
            // Interpolate Green at R/B location
            float g_avg = (raw.data[idx-1] + raw.data[idx+1] + 
                           raw.data[idx-w] + raw.data[idx+w]) / 4.0f;
            
            // Laplacian of the underlying color (R or B)
            // 2nd derivative: 4*Center - (N+S+E+W)
            float c_lap = 4.0f * raw.data[idx] - 
                         (raw.data[idx-2] + raw.data[idx+2] + 
                          raw.data[idx-2*w] + raw.data[idx+2*w]);
            
            // Correction term
            // G = G_bilinear + 0.5 * Laplacian(R/B)
            // Note: The formula varies slightly in literature. 
            // Often: G = G_bilinear + (R_center - R_bilinear_at_G_locs)
            // Simplified gradient correction:
            
            float val = g_avg + 0.125f * c_lap; 
            
            if (val < 0) val = 0;
            green_channel[idx] = val;
        }
    }
}
```

### Example 3: Edge-Directed Interpolation

Compute horizontal and vertical gradients. Pick the direction with smaller gradient.

```cpp
/**
 * @brief Edge-Directed Interpolation (Green)
 */
void demosaic_edge_directed(const RawImage& raw, float* green_channel) {
    int w = raw.width;
    
    #pragma omp parallel for
    for (int y = 2; y < raw.height - 2; y++) {
        for (int x = 2; x < raw.width - 2; x++) {
            int idx = y * w + x;
            
            if ((y + x) % 2 == 1) {
                green_channel[idx] = raw.data[idx];
                continue;
            }
            
            // Calculate Gradients
            float h_grad = abs(raw.data[idx-2] - raw.data[idx+2]);
            float v_grad = abs(raw.data[idx-2*w] - raw.data[idx+2*w]);
            
            float g_val;
            
            if (h_grad < v_grad) {
                // Horizontal edge: Interpolate horizontally
                g_val = (raw.data[idx-1] + raw.data[idx+1]) / 2.0f;
            } else if (v_grad < h_grad) {
                // Vertical edge: Interpolate vertically
                g_val = (raw.data[idx-w] + raw.data[idx+w]) / 2.0f;
            } else {
                // No strong edge: Average all
                g_val = (raw.data[idx-1] + raw.data[idx+1] + 
                         raw.data[idx-w] + raw.data[idx+w]) / 4.0f;
            }
            
            green_channel[idx] = g_val;
        }
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Zipper" Hunt

**Objective:** Visualize zipper artifacts.

**Steps:**
1.  Capture a raw image of text or a high-contrast edge (black/white).
2.  Process with **Bilinear** interpolation.
3.  Zoom in (400%) on the edges.
4.  **Observation:** You should see a checkerboard/zipper pattern on the edge.
5.  Process with **Edge-Directed**.
6.  **Observation:** The zipper should be significantly reduced or gone.

### Lab 2: False Color Analysis

**Objective:** Observe false color in high-frequency patterns.

**Steps:**
1.  Capture a "Zone Plate" chart or a fine striped shirt.
2.  Process with Bilinear.
3.  **Observation:** You will see rainbows (Aliasing) in the high-frequency areas.
4.  **Challenge:** Try to implement a Median Filter on the Chroma channels (post-demosaic) to reduce this.

### Lab 3: Performance Benchmarking

**Objective:** Measure CPU cost.

**Steps:**
1.  Implement Bilinear and Malvar-He-Cutler.
2.  Run on a 12MP image (4000x3000).
3.  Measure time using `std::chrono`.
4.  **Optimization:** Use OpenMP (`#pragma omp parallel for`) and measure speedup.

---

## 🐛 Debugging Techniques

### Debug 1: Grid Pattern in Flat Areas

**Symptom:** Flat grey areas look like a grid or checkerboard.

**Cause:**
*   **Imbalance:** The Green pixels on Red rows (Gr) and Green pixels on Blue rows (Gb) have different sensitivities or crosstalk.
*   **Fix:** Apply "Gr/Gb Imbalance Correction" in the Raw domain *before* demosaicing. Average them or apply separate gains.

### Debug 2: Color Dots at Nyquist

**Symptom:** Single pixel dots of wrong color.

**Cause:**
*   Noise interpreted as detail.
*   **Fix:** Apply Raw Denoise before demosaicing.

---

## ⚡ Performance Optimization

### Optimization 1: Integer Math

Avoid floats.
*   `val = (a + b) / 2.0f` -> `val = (a + b) >> 1`
*   `val = 0.125 * x` -> `val = x >> 3`

### Optimization 2: SIMD (NEON/AVX)

*   Load 16 raw pixels.
*   De-interleave into vectors.
*   Perform arithmetic.
*   Store.
*   **Gain:** 4x-8x speedup over scalar C++.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is the Green channel sampled at a higher rate than Red/Blue?**
2.  **Explain the "Constant Hue" assumption.**
3.  **What causes Moiré patterns?**
4.  **Why does Bilinear interpolation fail at edges?**

### Practical Challenges

1.  **Implement the "Constant Hue" interpolation** for Red/Blue channels:
    *   Interpolate `(R - G)` instead of `R`.
    *   `R_at_G = G_at_G + Interpolate(R - G)`.
2.  **Create a synthetic Bayer image** from a high-res RGB image and test your demosaicing algorithm against the ground truth. Calculate PSNR.

---

## 📚 Further Reading & Resources

### Papers
*   **"High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color Images"** - Malvar, He, Cutler (Microsoft Research).
*   **"Adaptive Homogeneity-Directed Demosaicing Algorithm"** - Hirakawa, Parks.

### Code
*   **dcraw:** The reference implementation for many demosaicing algorithms (look at `ahd_interpolate`).

---

## 🎓 Summary

Today we covered:
- ✅ **Bayer CFA:** The structure of raw data.
- ✅ **Bilinear:** The baseline algorithm.
- ✅ **Edge-Directed:** Following gradients to preserve edges.
- ✅ **Artifacts:** Zippers and False Color.
- ✅ **Optimization:** Making it fast.

**Next:** Day 24 - Color Processing (CCM, Gamma, CSC).

---

**Day 23 Complete** | Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development
