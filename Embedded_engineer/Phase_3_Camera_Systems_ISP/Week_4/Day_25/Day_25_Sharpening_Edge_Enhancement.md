# Day 25: Sharpening & Edge Enhancement
## Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development

---

## 🎯 Learning Objectives
1.  **Understand** the concept of Acutance vs Resolution.
2.  **Implement** Unsharp Masking (USM) for basic sharpening.
3.  **Develop** Edge-Adaptive Sharpening to prevent halo artifacts.
4.  **Apply** High-Pass Filters (Laplacian) for detail extraction.
5.  **Tune** sharpening parameters (Strength, Radius, Threshold).
6.  **Debug** over-sharpening artifacts (Halos, Noise amplification).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera or Test Images (ISO 12233 Chart).
*   **Software:** C/C++ Compiler.
*   **Knowledge:** Convolution, Frequency Domain (High/Low frequencies).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Sharpen?
Lenses act as Low-Pass Filters (Optical Blur). The Anti-Aliasing (OLPF) filter and Demosaicing process further blur the image.
*   **Goal:** Restore the high-frequency components to make the image look "crisp".
*   **Acutance:** The subjective perception of sharpness (edge contrast).
*   **Resolution:** The actual detail resolving power (line pairs/mm).
*   **Sharpening increases Acutance, not Resolution.**

### 🔹 Part 2: Unsharp Masking (USM)
The classic algorithm from darkroom photography.
1.  **Blur** the original image (Gaussian Blur).
2.  **Subtract** the blurred version from the original. This gives the "Mask" (High Frequencies).
3.  **Add** the Mask back to the original.
    *   `Sharp = Original + Strength * (Original - Blurred)`

### 🔹 Part 3: The Halo Problem
If you sharpen too much, you get "Halos" (bright lines along dark edges).
*   **Overshoot:** The edge transition goes above white or below black.
*   **Solution:** Edge-Adaptive Sharpening. We limit the sharpening amount based on the local edge strength or brightness.

---

## 💻 Implementation Examples

### Example 1: Basic Unsharp Mask (USM)

```cpp
/**
 * @brief Apply Unsharp Mask
 * @param img Y Channel (Luma)
 * @param strength Amount of sharpening (0.0 to 5.0)
 */
void apply_usm(Image8& img, float strength) {
    int w = img.width;
    int h = img.height;
    std::vector<uint8_t> blurred(w * h);
    
    // 1. Apply Gaussian Blur (3x3 or 5x5)
    // Simplified 3x3 box blur for demo
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += img.data[(y+dy)*w + (x+dx)];
                }
            }
            blurred[y*w + x] = sum / 9;
        }
    }
    
    // 2. Add High Frequency back
    #pragma omp parallel for
    for (int i = 0; i < w * h; i++) {
        int orig = img.data[i];
        int blur = blurred[i];
        int mask = orig - blur; // High Pass
        
        int sharp = orig + (int)(mask * strength);
        
        img.data[i] = (uint8_t)std::clamp(sharp, 0, 255);
    }
}
```

### Example 2: Laplacian Sharpening (Convolution)

Using a single convolution kernel to extract edges and add them.
Kernel:
```
 0 -1  0
-1  5 -1
 0 -1  0
```
This kernel effectively does `Original + (Original - Average)`.

```cpp
/**
 * @brief Apply Laplacian Sharpening
 */
void apply_laplacian(Image8& img) {
    int w = img.width;
    int h = img.height;
    std::vector<uint8_t> output(w * h);
    
    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += img.data[(y+dy)*w + (x+dx)] * kernel[dy+1][dx+1];
                }
            }
            output[y*w + x] = (uint8_t)std::clamp(sum, 0, 255);
        }
    }
    
    // Copy back
    memcpy(img.data.data(), output.data(), w * h);
}
```

### Example 3: Edge-Adaptive Sharpening (Halo Control)

We clamp the sharpening amount so it doesn't exceed the local min/max of the neighborhood.

```cpp
/**
 * @brief Halo-Free Sharpening
 */
void apply_smart_sharpen(Image8& img, float strength) {
    int w = img.width;
    int h = img.height;
    std::vector<uint8_t> output(w * h);
    
    #pragma omp parallel for
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            // 1. Find Local Min/Max
            uint8_t local_min = 255;
            uint8_t local_max = 0;
            int blur_sum = 0;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    uint8_t val = img.data[(y+dy)*w + (x+dx)];
                    if (val < local_min) local_min = val;
                    if (val > local_max) local_max = val;
                    blur_sum += val;
                }
            }
            int blur = blur_sum / 9;
            int orig = img.data[y*w + x];
            
            // 2. Calculate Unsharp Mask
            int high_pass = orig - blur;
            int sharp = orig + (int)(high_pass * strength);
            
            // 3. Clamp to prevent Halos (Overshoot)
            // Allow a small overshoot (e.g., 10%) but not too much
            int overshoot = (local_max - local_min) / 10;
            
            if (sharp > local_max + overshoot) sharp = local_max + overshoot;
            if (sharp < local_min - overshoot) sharp = local_min - overshoot;
            
            output[y*w + x] = (uint8_t)std::clamp(sharp, 0, 255);
        }
    }
    memcpy(img.data.data(), output.data(), w * h);
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Tuning the USM

**Objective:** Find the "Sweet Spot" for sharpening.

**Steps:**
1.  Take an image of a face (soft details) and text (hard edges).
2.  Apply USM with Strength = 0.5, 1.0, 2.0, 4.0.
3.  **Observation:**
    *   0.5: Subtle pop.
    *   1.0: Crisp.
    *   2.0: Halos appear. Skin looks rough (noise amplified).
    *   4.0: Cartoonish artifacts.

### Lab 2: Frequency Separation

**Objective:** Sharpen only fine details, not coarse shapes.

**Steps:**
1.  Create two blurred versions: `Blur_Small` (Radius 1) and `Blur_Large` (Radius 5).
2.  `Detail = Blur_Small - Blur_Large`. (Band-pass filter).
3.  Add `Detail` to original.
4.  **Result:** Enhances textures without creating thick halos on main object boundaries.

### Lab 3: Noise Masking

**Objective:** Don't sharpen noise.

**Steps:**
1.  Calculate a "Flat Area" mask (Variance map).
2.  If Variance < Threshold, set Sharpen Strength = 0.
3.  If Variance > Threshold, set Sharpen Strength = 1.0.
4.  **Result:** Smooth sky (no noise), sharp buildings.

---

## 🐛 Debugging Techniques

### Debug 1: "Worms" in Flat Areas

**Symptom:** Squiggly lines in the sky or walls.

**Cause:** Sharpening noise.
*   **Fix:** Increase the "Threshold" parameter (Corresponds to Lab 3). Only sharpen if `abs(Original - Blur) > Threshold`.

### Debug 2: Jagged Diagonals (Aliasing)

**Symptom:** Diagonal lines look like stairs.

**Cause:** Over-sharpening already aliased edges.
*   **Fix:** Use a "Directional" sharpener that sharpens *along* the edge, not across it (similar to edge-directed demosaicing).

---

## ⚡ Performance Optimization

### Optimization 1: Separable Blur

Gaussian Blur is separable.
*   Instead of 2D convolution (NxN), do 1D Horizontal (N) then 1D Vertical (N).
*   Complexity reduces from $O(N^2)$ to $O(2N)$.

### Optimization 2: Fixed Point

*   Sharpening strength is usually a float (e.g., 1.5).
*   Use Q8 format: `Strength_Int = 384` (1.5 * 256).
*   `Sharp = Orig + ((Mask * Strength_Int) >> 8)`.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Does sharpening add information to the image?**
2.  **Why do we sharpen Luma (Y) instead of RGB?**
3.  **What is the relationship between Gaussian Blur radius and the frequency enhanced?**
4.  **How does "Threshold" prevent noise amplification?**

### Practical Challenges

1.  **Implement "Unsharp Mask" using the Difference of Gaussians (DoG)** method.
2.  **Create a "Clarity" filter** (Local Contrast Enhancement) using a large radius USM.

---

## 📚 Further Reading & Resources

### Papers
*   **"Adaptive Unsharp Masking for Contrast Enhancement"** - Polesel et al.

### Tools
*   **Photoshop:** Play with "Unsharp Mask" and "Smart Sharpen" to understand parameters.

---

## 🎓 Summary

Today we covered:
- ✅ **Sharpening Physics:** Restoring acutance.
- ✅ **USM:** The workhorse algorithm.
- ✅ **Artifacts:** Halos and Noise.
- ✅ **Adaptive Methods:** Clamping and Thresholding.
- ✅ **Tuning:** Balancing crispness vs natural look.

**Next:** Day 26 - Noise Reduction (Spatial & Temporal).

---

**Day 25 Complete** | Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development
