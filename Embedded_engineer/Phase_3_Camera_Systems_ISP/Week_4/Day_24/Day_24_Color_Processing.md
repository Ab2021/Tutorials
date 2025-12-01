# Day 24: Color Processing (CCM, Gamma, CSC)
## Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development

---

## 🎯 Learning Objectives
1.  **Understand** the physics of color reproduction and the need for color correction.
2.  **Implement** the Color Correction Matrix (CCM) to map sensor RGB to sRGB/Rec.709.
3.  **Apply** Gamma Correction to handle non-linear display characteristics.
4.  **Perform** Color Space Conversions (RGB to YUV/YCbCr) for compression.
5.  **Develop** 3D Look-Up Table (LUT) support for advanced color grading.
6.  **Debug** color artifacts (saturation clipping, hue shifts, banding).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with Raw output, Macbeth ColorChecker Chart (standard 24-patch).
*   **Software:** C/C++ Compiler, Python (NumPy/SciPy for calibration).
*   **Knowledge:** Linear Algebra (Matrix multiplication), Colorimetry (CIE 1931 XYZ).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Color Pipeline

After demosaicing, we have a full RGB image, but the colors are "Device Dependent."
*   **Sensor RGB:** Defined by the spectral sensitivity of the silicon photodiodes and the CFA filters.
*   **Target RGB:** Usually sRGB (Standard RGB) or Rec.709 (HDTV).

**The Transformation Chain:**
`Sensor RGB -> [AWB Gain] -> [CCM] -> [Gamma] -> sRGB -> [CSC] -> YUV`

### 🔹 Part 2: Color Correction Matrix (CCM)

Sensors are not colorimetric. Their spectral response does not match the human eye (LMS cones).
*   **The Problem:** A sensor might see "Red" as slightly orange or "Blue" as slightly purple compared to a standard observer.
*   **The Solution:** A 3x3 Matrix multiplication to mix channels.
    ```
    | R_out |   | c00 c01 c02 |   | R_in |
    | G_out | = | c10 c11 c12 | * | G_in |
    | B_out |   | c20 c21 c22 |   | B_in |
    ```
*   **Calibration:** We minimize the error between the *captured* colors of a Macbeth chart and the *known* reference values (Lab/XYZ) using Least Squares optimization.

### 🔹 Part 3: Gamma Correction

Human vision is logarithmic (we are more sensitive to changes in dark tones). Displays (CRTs) had a power-law response.
*   **Linear RGB:** Sensor data is linear (photons count).
*   **Gamma Encoding:** We apply a power function (usually $\gamma \approx 2.2$) to compress bright tones and expand dark tones.
    *   $V_{out} = V_{in}^{(1/\gamma)}$
*   **sRGB Transfer Function:** A specific curve with a linear section near black and a 2.4 power curve elsewhere.

### 🔹 Part 4: Color Space Conversion (CSC)

For compression (JPEG/H.264), we separate Luma (Brightness) from Chroma (Color).
*   **Y:** Luma (Weighted sum of R, G, B).
*   **U/V (or Cb/Cr):** Color difference signals ($B-Y$, $R-Y$).
*   **Subsampling:** We can reduce resolution of U/V (4:2:0 or 4:2:2) because the eye is less sensitive to color detail.

---

## 💻 Implementation Examples

### Example 1: Applying CCM (C++)

Optimized 3x3 matrix multiplication.

```cpp
/**
 * @brief Apply Color Correction Matrix
 * @param input Linear RGB (0-1.0 range or 0-1023)
 * @param ccm 3x3 Matrix (Row-major)
 */
void apply_ccm(RGBImage& img, const float* ccm) {
    int w = img.width;
    int h = img.height;
    
    #pragma omp parallel for
    for (int i = 0; i < w * h; i++) {
        float r = img.data[i].r;
        float g = img.data[i].g;
        float b = img.data[i].b;
        
        float r_new = ccm[0]*r + ccm[1]*g + ccm[2]*b;
        float g_new = ccm[3]*r + ccm[4]*g + ccm[5]*b;
        float b_new = ccm[6]*r + ccm[7]*g + ccm[8]*b;
        
        // Clamp to valid range (e.g., 0-1023)
        // Note: CCM can produce negative values!
        img.data[i].r = (uint16_t)std::max(0.0f, std::min(1023.0f, r_new));
        img.data[i].g = (uint16_t)std::max(0.0f, std::min(1023.0f, g_new));
        img.data[i].b = (uint16_t)std::max(0.0f, std::min(1023.0f, b_new));
    }
}
```

### Example 2: Gamma Correction (LUT Based)

Power functions are slow (`powf`). We use a Look-Up Table (LUT).

```cpp
/**
 * @brief Initialize Gamma LUT
 * @param gamma Typically 2.2
 * @param max_val Input max value (e.g., 1023)
 */
std::vector<uint8_t> init_gamma_lut(float gamma, int max_val) {
    std::vector<uint8_t> lut(max_val + 1);
    float inv_gamma = 1.0f / gamma;
    
    for (int i = 0; i <= max_val; i++) {
        float norm = (float)i / max_val;
        float val = powf(norm, inv_gamma) * 255.0f; // Output 8-bit
        lut[i] = (uint8_t)std::min(255.0f, std::max(0.0f, val));
    }
    return lut;
}

/**
 * @brief Apply Gamma LUT
 */
void apply_gamma(RGBImage& img, const std::vector<uint8_t>& lut) {
    int size = img.width * img.height;
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        // Convert 10/12-bit linear to 8-bit gamma corrected
        img.data[i].r = lut[img.data[i].r];
        img.data[i].g = lut[img.data[i].g];
        img.data[i].b = lut[img.data[i].b];
    }
}
```

### Example 3: RGB to YUV444 Conversion (BT.601)

Standard definition matrix.
$Y = 0.299R + 0.587G + 0.114B$
$U = -0.147R - 0.289G + 0.436B$
$V = 0.615R - 0.515G - 0.100B$

```cpp
/**
 * @brief RGB to YUV Conversion (Integer Math)
 * Input: 8-bit RGB
 * Output: 8-bit YUV
 */
void rgb_to_yuv(const RGBImage8& rgb, YUVImage& yuv) {
    int size = rgb.width * rgb.height;
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        int r = rgb.data[i].r;
        int g = rgb.data[i].g;
        int b = rgb.data[i].b;
        
        // Using Q8 fixed point (multiply by 256)
        // Y
        int y = (77 * r + 150 * g + 29 * b) >> 8;
        
        // U (Cb) - Offset by 128
        int u = ((-43 * r - 84 * g + 127 * b) >> 8) + 128;
        
        // V (Cr) - Offset by 128
        int v = ((127 * r - 106 * g - 21 * b) >> 8) + 128;
        
        yuv.data[i].y = (uint8_t)std::clamp(y, 0, 255);
        yuv.data[i].u = (uint8_t)std::clamp(u, 0, 255);
        yuv.data[i].v = (uint8_t)std::clamp(v, 0, 255);
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: CCM Calibration (Python)

**Objective:** Calculate the optimal CCM for your camera.

**Steps:**
1.  Capture a Raw image of a Macbeth ColorChecker.
2.  Extract the average RGB values for the 24 patches.
3.  Load standard sRGB reference values for the patches.
4.  **Optimization:**
    *   Use `scipy.optimize.least_squares`.
    *   Target Function: `minimize || (CCM * SensorRGB) - ReferenceRGB ||`.
    *   Constraint: Row sums should equal 1 (to preserve white balance).
5.  **Result:** A 3x3 matrix. Apply it to your C++ pipeline and observe the colors become "correct".

### Lab 2: Gamma Curve Visualization

**Objective:** See the effect of Gamma.

**Steps:**
1.  Create a linear gradient image (0 to 255).
2.  Display it on your monitor. It will look dark in the middle.
3.  Apply Gamma 2.2.
4.  Display it. It should look perceptually linear (smooth transition).

### Lab 3: 3D LUT Implementation

**Objective:** Implement advanced color grading.

**Steps:**
1.  Create a 17x17x17 3D LUT (Cube).
2.  Populate it with an Identity transform (or a "Teal and Orange" look).
3.  **Algorithm:**
    *   For each pixel (r,g,b), find the cube cell.
    *   Perform Trilinear Interpolation within the cell.
4.  Apply to an image.

---

## 🐛 Debugging Techniques

### Debug 1: Color Banding (Posterization)

**Symptom:** Smooth gradients look like steps.

**Cause:**
*   Applying heavy processing (Gamma/CCM) on 8-bit data.
*   **Fix:** Keep data in 10-bit, 12-bit, or Float as long as possible. Only convert to 8-bit at the very end.

### Debug 2: Saturation Clipping

**Symptom:** Bright colors turn white or wrong hue (e.g., bright blue turns cyan).

**Cause:**
*   CCM boosts a channel beyond the max value.
*   Simple clamping `min(val, max)` desaturates the color.
*   **Fix:** Use "Gamut Mapping" or "Desaturation" algorithms that preserve Hue when Luma is high.

---

## ⚡ Performance Optimization

### Optimization 1: NEON Intrinsics for CCM

Matrix multiplication is heavy.
*   Load R, G, B into vectors.
*   Use `vmla` (Vector Multiply Accumulate).
*   Process 4 or 8 pixels per cycle.

### Optimization 2: YUV Subsampling

*   Instead of calculating U/V for every pixel, calculate for every 2x2 block (average inputs).
*   Saves 75% of chroma calculations.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is the CCM usually applied on Linear data, not Gamma corrected data?**
2.  **What is the difference between sRGB and Rec.709?**
3.  **Why do we add 128 to U and V channels?**
4.  **How does a 3D LUT differ from three 1D LUTs?**

### Practical Challenges

1.  **Implement a "Saturation" slider** in your ISP.
    *   Convert RGB -> HSV -> Scale S -> RGB.
    *   Or use a Matrix approach: `M_sat = Interpolate(Identity, Grayscale, t)`.
2.  **Optimize the Gamma LUT** to use 12-bit input (4096 entries) and 8-bit output.

---

## 📚 Further Reading & Resources

### Standards
*   **IEC 61966-2-1:** sRGB Standard.
*   **ITU-R BT.709:** HDTV Standards.

### Tools
*   **Imatest:** Industry standard for CCM calibration.
*   **dcamprof:** Open source camera profiling tool.

---

## 🎓 Summary

Today we covered:
- ✅ **Color Pipeline:** The sequence of transforms.
- ✅ **CCM:** Mapping sensor colors to reality.
- ✅ **Gamma:** Matching the display response.
- ✅ **CSC:** Preparing for video compression.
- ✅ **LUTs:** The tool for creative color grading.

**Next:** Day 25 - Sharpening & Edge Enhancement.

---

**Day 24 Complete** | Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development
