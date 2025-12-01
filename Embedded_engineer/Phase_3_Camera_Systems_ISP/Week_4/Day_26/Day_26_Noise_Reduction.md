# Day 26: Noise Reduction (Spatial & Temporal)
## Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development

---

## 🎯 Learning Objectives
1.  **Analyze** the sources of noise (Photon Shot Noise, Read Noise, Fixed Pattern Noise).
2.  **Implement** Spatial Denoising: Bilateral Filter (Edge-Preserving).
3.  **Implement** Temporal Denoising: Motion-Adaptive Averaging (3DNR).
4.  **Compare** algorithms: Gaussian vs Median vs Non-Local Means (NLM).
5.  **Tune** noise reduction strength based on ISO gain.
6.  **Debug** artifacts: Ghosting (Temporal) and "Oil Painting" effect (Spatial).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera capturing low-light video.
*   **Software:** C/C++ Compiler.
*   **Knowledge:** Probability (Gaussian Distribution), Signal-to-Noise Ratio (SNR).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Noise Physics
*   **Photon Shot Noise:** Random fluctuation of photon arrival. Follows Poisson distribution. Dominant in bright areas. $\sigma \propto \sqrt{Signal}$.
*   **Read Noise:** Electronic noise from the sensor circuitry. Gaussian. Dominant in dark areas.
*   **FPN (Fixed Pattern Noise):** Pixel-to-pixel sensitivity variations. Removed by DPC/BLC.

### 🔹 Part 2: Spatial Denoising (2D)
Filtering within a single frame.
*   **Gaussian Blur:** Averages neighbors. Removes noise but blurs edges.
*   **Median Filter:** Replaces pixel with median of neighbors. Good for "Salt & Pepper" noise.
*   **Bilateral Filter:** The gold standard for basic ISP.
    *   Weights neighbors by **Distance** (how far away) AND **Intensity Difference** (how different the color is).
    *   If a neighbor is far away OR has a very different color (edge), it contributes less.
    *   Result: Smooths flat areas, preserves edges.

### 🔹 Part 3: Temporal Denoising (3D / TNR)
Filtering across multiple frames.
*   **Concept:** Noise is random across time. Signal is constant (for static scenes). Averaging $N$ frames reduces noise variance by $N$.
*   **Motion Handling:** If the scene moves, averaging causes "Ghosting" (trails).
*   **Motion Adaptive:** Detect motion. If moving, use Spatial NR. If static, use Temporal NR.

---

## 💻 Implementation Examples

### Example 1: Bilateral Filter (Spatial)

```cpp
/**
 * @brief Apply Bilateral Filter
 * @param sigma_s Spatial Sigma (Distance)
 * @param sigma_r Range Sigma (Intensity)
 */
void apply_bilateral(Image8& img, float sigma_s, float sigma_r) {
    int w = img.width;
    int h = img.height;
    std::vector<uint8_t> output(w * h);
    
    int r = (int)ceil(2.0f * sigma_s); // Kernel radius
    
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0;
            float weight_sum = 0;
            int center_val = img.data[y*w + x];
            
            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    
                    // Boundary check
                    if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                    
                    int neighbor_val = img.data[ny*w + nx];
                    
                    // Spatial Weight (Gaussian)
                    float dist_sq = (float)(dx*dx + dy*dy);
                    float w_s = expf(-dist_sq / (2 * sigma_s * sigma_s));
                    
                    // Range Weight (Gaussian)
                    float diff = (float)(neighbor_val - center_val);
                    float w_r = expf(-(diff*diff) / (2 * sigma_r * sigma_r));
                    
                    float weight = w_s * w_r;
                    
                    sum += neighbor_val * weight;
                    weight_sum += weight;
                }
            }
            output[y*w + x] = (uint8_t)(sum / weight_sum);
        }
    }
    memcpy(img.data.data(), output.data(), w * h);
}
```

### Example 2: Simple Temporal NR (IIR Filter)

Infinite Impulse Response filter. Blends current frame with previous frame.
`Out[t] = Alpha * In[t] + (1 - Alpha) * Out[t-1]`

```cpp
/**
 * @brief Apply Temporal Noise Reduction (Motion Adaptive)
 * @param curr Current Frame
 * @param prev Previous Output Frame (Accumulator)
 */
void apply_tnr(Image8& curr, Image8& prev) {
    int size = curr.width * curr.height;
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        int c = curr.data[i];
        int p = prev.data[i];
        
        // 1. Motion Detection
        // Simple absolute difference
        int diff = abs(c - p);
        
        // 2. Calculate Blending Factor (Alpha)
        // If diff is high (Motion), Alpha -> 1.0 (Use Current, No History)
        // If diff is low (Static), Alpha -> 0.1 (Heavy Averaging)
        
        float alpha;
        if (diff > 20) {
            alpha = 1.0f; // Motion: No TNR
        } else if (diff > 10) {
            alpha = 0.5f; // Slight Motion: Weak TNR
        } else {
            alpha = 0.1f; // Static: Strong TNR
        }
        
        // 3. Blend
        int out = (int)(alpha * c + (1.0f - alpha) * p);
        
        // Update History
        prev.data[i] = (uint8_t)out;
        // Output is also written to curr for display
        curr.data[i] = (uint8_t)out;
    }
}
```

### Example 3: Non-Local Means (NLM) - Concept

Instead of comparing single pixels (like Bilateral), NLM compares **Patches**.
*   If Patch A looks like Patch B, average the center pixels.
*   Computationally expensive but excellent quality.
*   Often implemented on GPU or dedicated HW.

---

## 🔬 Hands-On Lab Exercises

### Lab 1: ISO vs Noise Profile

**Objective:** Characterize your sensor's noise.

**Steps:**
1.  Set Camera to ISO 100. Capture a Grey Card.
2.  Calculate Standard Deviation ($\sigma$) of a 100x100 patch.
3.  Repeat for ISO 200, 400, 800, 1600, 3200.
4.  **Plot:** ISO vs $\sigma$.
5.  **Usage:** Use this table to tune your `sigma_r` in Bilateral Filter dynamically based on current ISO.

### Lab 2: The "Oil Painting" Effect

**Objective:** Observe over-denoising.

**Steps:**
1.  Take a noisy image (ISO 3200).
2.  Apply Bilateral Filter with very high `sigma_s` (10.0) and `sigma_r` (50.0).
3.  **Observation:** Textures (skin pores, fabric) are wiped out. The image looks flat and waxy.
4.  **Tuning:** Reduce `sigma_r` until textures return, even if some noise remains.

### Lab 3: Ghosting Hunt

**Objective:** Debug TNR artifacts.

**Steps:**
1.  Enable TNR (Example 2).
2.  Wave your hand in front of the camera.
3.  Set `alpha = 0.05` (Strong averaging) and disable motion detection (always use 0.05).
4.  **Observation:** You will see a long trail behind your hand.
5.  Re-enable Motion Detection. The trail should disappear.

---

## 🐛 Debugging Techniques

### Debug 1: "Floating Dust" (Dirty Window Effect)

**Symptom:** Noise pattern stays static while the scene moves.

**Cause:** FPN (Fixed Pattern Noise) is not corrected. TNR preserves it because it looks "Static".
*   **Fix:** Better DPC/BLC calibration. Or subtract a "Dark Frame" before TNR.

### Debug 2: Texture Breathing

**Symptom:** Textures disappear and reappear randomly.

**Cause:** Motion detection threshold is toggling on noise.
*   **Fix:** Use Hysteresis for the motion threshold. Or use a "Soft Switch" (Sigmoid function) for Alpha instead of hard `if/else`.

---

## ⚡ Performance Optimization

### Optimization 1: Fast Bilateral (Grid)

*   Approximating Bilateral Filter using a 3D Grid (x, y, intensity).
*   Downsample -> Filter -> Upsample.
*   Reduces complexity from $O(r^2)$ to $O(1)$.

### Optimization 2: Chroma Subsampling for NR

*   The eye is less sensitive to Chroma noise.
*   Apply Strong NR on UV channels (Aggressive Bilateral).
*   Apply Weak NR on Y channel (Preserve texture).
*   Perform UV NR at 1/2 resolution (4:2:0).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does averaging frames reduce noise?**
2.  **What is the main advantage of Bilateral Filter over Gaussian Blur?**
3.  **Why do we need different tuning parameters for high ISO?**
4.  **How does "Read Noise" differ from "Shot Noise"?**

### Practical Challenges

1.  **Implement a "Luma-Adaptive" Bilateral Filter:**
    *   Dark areas have more noise (low SNR). Increase `sigma_r` for dark pixels.
    *   Bright areas have less noise. Decrease `sigma_r`.
2.  **Create a "Chroma Only" denoiser:** Convert RGB->YUV, blur UV, convert back.

---

## 📚 Further Reading & Resources

### Papers
*   **"Bilateral Filtering for Gray and Color Images"** - Tomasi & Manduchi.
*   **"Non-Local Means Denoising"** - Buades et al.

### Tools
*   **BM3D:** The state-of-the-art software denoising algorithm (Benchmark).

---

## 🎓 Summary

Today we covered:
- ✅ **Noise Types:** Shot, Read, FPN.
- ✅ **Spatial NR:** Bilateral Filter for edge-preserving smoothing.
- ✅ **Temporal NR:** Using history to reduce variance.
- ✅ **Motion Adaptation:** Preventing ghosting.
- ✅ **Tuning:** ISO-based parameter maps.

**Next:** Day 27 - Week 4 Review & ISP Tuning Project.

---

**Day 26 Complete** | Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development
