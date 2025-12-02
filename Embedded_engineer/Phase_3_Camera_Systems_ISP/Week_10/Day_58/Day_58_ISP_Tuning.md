# Day 58: ISP Tuning Workflow (The Art of Tuning)
## Phase 3: Camera Systems & ISP | Week 10: Manufacturing, Calibration & Tuning

---

## 🎯 Learning Objectives
1.  **Understand** the logical order of ISP tuning (Linear Domain -> Color Domain -> YUV Domain).
2.  **Tune** the Black Level Correction (BLC) and Lens Shading (LSC).
3.  **Calibrate** the Color Correction Matrix (CCM) for accurate color reproduction.
4.  **Adjust** Gamma and Tone Mapping for contrast.
5.  **Optimize** Sharpening and Noise Reduction (NR) for different light levels (Low/Mid/High Lux).
6.  **Evaluate** Image Quality (IQ) subjectively and objectively.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with Tunable ISP (Raspberry Pi, Rockchip, Ambarella).
*   **Software:** ISP Tuning Tool (e.g., Raspberry Pi Tuning Tool, IQTools).
*   **Knowledge:** Color Spaces, Noise Models.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Tuning Pipeline
Order matters! You cannot tune Color before Black Level.
1.  **Linear Domain:** BLC -> LSC -> AWB -> Demosaic.
2.  **Color Domain:** CCM -> Gamma -> 3D LUT.
3.  **YUV Domain:** Sharpening -> Noise Reduction -> Scaling.

### 🔹 Part 2: Color Tuning (CCM)
*   **Goal:** Make the camera colors match reality (or a preference).
*   **Method:** Capture a Macbeth ColorChecker (24 patches). Minimize the DeltaE error between the captured RGB values and the known reference values.
*   **Matrix:** $3 \times 3$ matrix.
    $$ \begin{bmatrix} R' \\ G' \\ B' \end{bmatrix} = \begin{bmatrix} c_{11} & c_{12} & c_{13} \\ c_{21} & c_{22} & c_{23} \\ c_{31} & c_{32} & c_{33} \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix} $$

### 🔹 Part 3: Noise & Sharpness Tuning
*   **Conflict:** Sharpening amplifies noise. NR blurs details.
*   **Strategy:**
    *   **High Light (Day):** Low NR, High Sharpening.
    *   **Low Light (Night):** High NR, Low Sharpening (to avoid "worms").
*   **Trigger:** Tuning parameters are interpolated based on Analog Gain (ISO).

---

## 💻 Implementation Examples

### Example 1: Calculating CCM (Python)

Finding the optimal matrix to minimize color error.

```python
import numpy as np
from scipy.optimize import minimize

# 1. Reference Colors (sRGB) for Macbeth Chart (24x3)
ref_srgb = np.array([[115, 82, 68], [194, 150, 130], ...]) / 255.0

# 2. Captured Colors (Linear RGB from Sensor)
cam_rgb = np.array([[0.1, 0.05, 0.04], [0.3, 0.2, 0.15], ...])

def error_func(matrix_flat):
    matrix = matrix_flat.reshape((3, 3))
    # Apply Matrix
    corrected = np.dot(cam_rgb, matrix.T)
    # Simple MSE (In reality, convert to Lab space for DeltaE)
    return np.mean((corrected - ref_srgb)**2)

# 3. Optimize
res = minimize(error_func, x0=np.eye(3).flatten())
ccm = res.x.reshape((3, 3))

print("Optimal CCM:\n", ccm)
```

### Example 2: Gamma Curve Generation

Creating a standard Gamma 2.2 LUT.

```cpp
void generate_gamma_lut(uint16_t* lut, int size) {
    for (int i = 0; i < size; i++) {
        float input = (float)i / (size - 1); // 0.0 to 1.0
        float output = pow(input, 1.0 / 2.2); // Gamma Correction
        lut[i] = (uint16_t)(output * (size - 1));
    }
}
```

### Example 3: Dynamic Tuning Logic (Pseudo-Code)

How the ISP driver applies settings at runtime.

```cpp
void update_isp_params(float analog_gain) {
    // Interpolate between Low Gain (1.0) and High Gain (16.0)
    float ratio = (analog_gain - 1.0) / (16.0 - 1.0);
    ratio = clamp(ratio, 0.0, 1.0);
    
    // Sharpening Strength
    float sharp_str = lerp(2.0, 0.5, ratio); // High -> Low
    isp_set_sharpening(sharp_str);
    
    // Noise Reduction Strength
    float nr_str = lerp(1.0, 5.0, ratio); // Low -> High
    isp_set_denoise(nr_str);
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Black Level Calibration

**Objective:** Set the foundation.

**Steps:**
1.  Cover lens (Dark).
2.  Capture Raw frame.
3.  Measure mean value (e.g., 64 in 10-bit).
4.  Set `BLC_Level = 64`.
5.  **Verify:** Capture again. Mean should be 0 (after BLC subtraction).

### Lab 2: Color Checker Tuning

**Objective:** Fix "Flat" colors.

**Steps:**
1.  Capture Macbeth chart under D65 (Daylight).
2.  Observe: Colors look washed out (low saturation) because sensor RGB is not sRGB.
3.  Run CCM optimization (Example 1).
4.  Apply Matrix.
5.  **Result:** Colors pop and match the chart.

### Lab 3: Sharpening Halo Test

**Objective:** Avoid over-sharpening.

**Steps:**
1.  Capture a high-contrast edge (Black text on White paper).
2.  Increase Sharpening Strength.
3.  **Observation:** A white line appears inside the black text (Overshoot) and a black line outside (Undershoot). This is "Halo".
4.  **Tune:** Reduce strength until Halo is barely visible but edge is crisp.

---

## 🐛 Debugging Tuning Issues

### Debug 1: "Pink Highlights"

**Symptom:** Bright white objects (clouds) turn pink.

**Cause:**
*   Green channel clips before Red/Blue.
*   AWB gains ($G_{gain} < R_{gain}$) push Red higher than Green in highlights.
*   **Fix:** Adjust the "White Level" clip point or implement "Highlight Recovery" (Luma-based desaturation).

### Debug 2: "Color Shading" in Corners

**Symptom:** Center is white, corners are green/purple.

**Cause:**
*   LSC (Lens Shading Correction) is weak or wrong color temperature.
*   **Fix:** Re-calibrate LSC. Ensure LSC strength is 100%.

---

## ⚡ Performance Optimization

### Optimization 1: Region-Based Tuning

*   Faces are more important than grass.
*   Detect faces and apply different tuning (Lower Sharpening, Skin Tone enhancement) to the face ROI.

### Optimization 2: 3D LUTs

*   Instead of simple Matrix + Gamma, use a 3D LUT (17x17x17 cube).
*   Allows complex non-linear color mapping (e.g., "Teal and Orange" look) efficiently.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do we need different CCMs for Daylight (D65) and Tungsten (A)?** (Sensor spectral response interacts with illuminant spectrum).
2.  **What is "Demosaicing Artifacts" (Zipper effect)?**
3.  **Why does Gamma correction come *after* CCM?** (CCM is a linear operation).
4.  **What is "Chroma Noise" vs "Luma Noise"?**

### Practical Challenges

1.  **Tune a "Cinematic" Look:** Create a tuning profile with low saturation, high contrast (S-Curve Gamma), and warm white balance.
2.  **Implement "Purple Fringe Removal":** Detect high-contrast edges with purple hue and desaturate them.

---

## 📚 Further Reading & Resources

### Tools
*   **Raspberry Pi Camera Tuning Guide.**
*   **Imatest Master:** For automated IQ analysis.

---

## 🎓 Summary

Today we covered:
- ✅ **Workflow:** Linear -> Color -> YUV.
- ✅ **BLC:** The absolute zero.
- ✅ **CCM:** Matching reality.
- ✅ **Gamma:** Managing contrast.
- ✅ **Sharpening/NR:** The eternal trade-off.

**Next:** Day 59 - Week 10 Review & Tuning Project.

---

**Day 58 Complete** | Phase 3: Camera Systems & ISP | Week 10: Manufacturing, Calibration & Tuning
