# Day 32: High Dynamic Range (HDR) Control
## Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control

---

## 🎯 Learning Objectives
1.  **Understand** the limitations of standard sensors (Dynamic Range).
2.  **Implement** Multi-Exposure HDR (Bracketing) control logic.
3.  **Analyze** Digital Overlap (DOL-HDR) sensors and their timing requirements.
4.  **Develop** Exposure Ratio Control (Long vs Short exposure).
5.  **Perform** Basic Tone Mapping (Global vs Local) to display HDR on LDR screens.
6.  **Debug** HDR artifacts (Motion blur, Ghosting, Color banding).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** HDR-capable sensor (e.g., IMX290, AR0231) or standard camera.
*   **Software:** C/C++ Compiler, OpenCV (for merging).
*   **Knowledge:** Radiometry, Logarithmic perception.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Dynamic Range Problem
*   **Real World:** Sunlight (100,000 lux) vs Shadow (100 lux) = 1000:1 ratio (60dB).
*   **Standard Sensor:** 10-bit or 12-bit ADC. Noise floor limits range to ~60-70dB.
*   **Result:** Either the sky is white (clipped) or the shadow is black (noise).

### 🔹 Part 2: HDR Techniques

#### 2.1 Multi-Exposure (Bracketing)
*   **Method:** Capture 3 frames sequentially: Short (Dark), Medium, Long (Bright).
*   **Merge:** Combine them in software.
*   **Pros:** Works with any sensor. High quality.
*   **Cons:** Motion artifacts (ghosting) because frames are taken at different times.

#### 2.2 Digital Overlap (DOL-HDR) / Staggered HDR
*   **Method:** Sensor outputs Long, Medium, and Short lines *interleaved* within the same frame readout.
*   **Timing:** The "Short" exposure happens *during* the readout of the "Long" exposure.
*   **Pros:** Minimal time gap between exposures (reduced ghosting).
*   **Cons:** Requires specialized sensor and ISP support.

### 🔹 Part 3: Tone Mapping
An HDR image has 16-20 bits of data. A display has 8 bits.
*   **Global Tone Mapping:** Apply a curve (Gamma/Log) to the whole image. Preserves contrast but loses local detail.
*   **Local Tone Mapping:** Adapt the curve based on local brightness. Preserves detail (e.g., texture in shadow) but can look "unnatural" or "halo-y".

---

## 💻 Implementation Examples

### Example 1: Calculating HDR Exposure Ratios

We need to decide the exposure times for Long (L), Medium (M), and Short (S) frames.
Typical Ratio: 4x, 8x, or 16x.

```cpp
struct HdrExposure {
    int long_us;
    int short_us;
    float ratio; // e.g., 16.0
};

/**
 * @brief Calculate HDR Exposures
 * @param total_exposure Target exposure for the "Long" frame
 * @param ratio Desired dynamic range extension (e.g., 16x)
 */
HdrExposure calculate_hdr_exposure(float total_exposure, float ratio) {
    HdrExposure out;
    
    // 1. Long Exposure (Base)
    // Decompose total_exposure into Gain/Time (Day 28 logic)
    // Simplified:
    out.long_us = (int)total_exposure; 
    
    // 2. Short Exposure
    // Short = Long / Ratio
    out.short_us = (int)(out.long_us / ratio);
    
    // Constraints
    if (out.short_us < 10) out.short_us = 10; // Min hardware limit
    
    out.ratio = (float)out.long_us / out.short_us;
    
    return out;
}
```

### Example 2: Simple HDR Merge (Linear)

Combining Long and Short images into a high-bitdepth linear buffer.

```cpp
/**
 * @brief Merge Long and Short Exposures
 * @param long_img Saturated in highlights
 * @param short_img Noisy in shadows
 * @param ratio Exposure Ratio (Long/Short)
 * @param threshold Crossover point (e.g., 90% of max value)
 */
void merge_hdr_linear(const Image12& long_img, const Image12& short_img, 
                      Image16& out_img, float ratio, int threshold) {
    int size = long_img.width * long_img.height;
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        uint16_t l_val = long_img.data[i];
        uint16_t s_val = short_img.data[i];
        
        uint16_t merged;
        
        if (l_val < threshold) {
            // Long exposure is good (not saturated)
            // Use Long value directly
            merged = l_val;
        } else {
            // Long exposure is clipped. Use Short.
            // Scale Short value up by Ratio to match Long's scale
            merged = (uint16_t)(s_val * ratio);
        }
        
        out_img.data[i] = merged;
    }
}
```

### Example 3: Global Tone Mapping (Reinhard)

A simple operator to compress range.
$L_{out} = L_{in} / (1 + L_{in})$

```cpp
/**
 * @brief Reinhard Tone Mapping
 * @param hdr_img 16-bit Linear HDR
 * @param ldr_img 8-bit Display
 */
void tone_map_reinhard(const Image16& hdr_img, Image8& ldr_img) {
    int size = hdr_img.width * hdr_img.height;
    
    // Find Max Luminance for normalization (optional)
    float max_lum = 65535.0f; 
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        float lin = (float)hdr_img.data[i] / max_lum; // 0.0 to 1.0
        
        // Reinhard Operator
        float mapped = lin / (1.0f + lin);
        
        // Gamma Correction (to sRGB)
        mapped = powf(mapped, 1.0f/2.2f);
        
        ldr_img.data[i] = (uint8_t)(mapped * 255.0f);
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Bracketing Sequence

**Objective:** Capture the raw data for HDR.

**Steps:**
1.  Set camera to Manual Mode.
2.  Scene: Dark room with a bright window.
3.  Capture Frame 1: Exposure 10ms (Window blown out, Room visible).
4.  Capture Frame 2: Exposure 1ms (Window visible, Room black).
5.  **Calculate Ratio:** 10ms / 1ms = 10x.

### Lab 2: The "Ghost" Effect

**Objective:** Observe motion artifacts.

**Steps:**
1.  Use the Bracketing setup (Lab 1).
2.  Wave your hand during the sequence.
3.  Merge the images (Example 2).
4.  **Observation:** You will see two hands (one bright, one dark) or a semi-transparent hand.
5.  **Fix:** Advanced "De-Ghosting" algorithms (Optical Flow) are needed for non-DOL sensors.

### Lab 3: Tone Mapping Tuning

**Objective:** Compare Global vs Local.

**Steps:**
1.  Take the merged HDR image.
2.  Apply Global Tone Mapping (Gamma). Result: Flat contrast.
3.  Apply Local Tone Mapping (CLAHE - Contrast Limited Adaptive Histogram Equalization).
4.  **Result:** Textures pop, but noise might increase in shadows.

---

## 🐛 Debugging Techniques

### Debug 1: Color Banding at Crossover

**Symptom:** A visible line or color shift where the image switches from Long to Short exposure.

**Cause:**
*   **Linearity Mismatch:** The sensor response isn't perfectly linear. `Short * Ratio` doesn't exactly match `Long`.
*   **White Balance:** WB gains might differ slightly if applied before merge.
*   **Fix:** Apply "Cross-Fading" (Alpha Blending) around the threshold instead of a hard switch.

### Debug 2: Flicker in HDR Video

**Symptom:** Brightness pulses.

**Cause:**
*   The Exposure Ratio is changing frame-to-frame.
*   **Fix:** Dampen the Ratio change in the AE algorithm. Lock the ratio (e.g., fixed 16x) if possible.

---

## ⚡ Performance Optimization

### Optimization 1: Companding (Knee Function)

*   Instead of storing 20-bit linear data, use a Piecewise Linear (PWL) or Log curve to compress it to 12-bit *before* writing to memory.
*   Many automotive sensors output 12-bit compressed HDR directly.

### Optimization 2: Single-Pass Merge & Tone Map

*   Don't write the intermediate 16-bit image to memory.
*   Read Long/Short -> Merge in registers -> Tone Map -> Write 8-bit.
*   Saves 50% memory bandwidth.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does "Short Exposure" have more noise than "Long Exposure" (after scaling)?**
2.  **What is the benefit of DOL-HDR over traditional Bracketing?**
3.  **Why do we need Tone Mapping? Why not just display the HDR image?**
4.  **How does "Flare" affect HDR performance?**

### Practical Challenges

1.  **Implement a "Soft Merge" function:**
    *   If `Long > 0.8 * Max`, start blending Short.
    *   `Weight = (Long - 0.8*Max) / (0.2*Max)`.
    *   `Out = (1-W)*Long + W*(Short*Ratio)`.
2.  **Simulate a DOL-HDR readout:** Write a script that generates an image with alternating lines from Long and Short exposures.

---

## 📚 Further Reading & Resources

### Papers
*   **"High Dynamic Range Imaging: Acquisition, Display, and Image-Based Lighting"** - Reinhard et al.

### Sensors
*   **Sony IMX290 / IMX390:** Datasheets describing DOL-HDR timing.
*   **OnSemi AR0231:** Datasheet describing Linear HDR.

---

## 🎓 Summary

Today we covered:
- ✅ **Dynamic Range:** The physical limits.
- ✅ **Multi-Exposure:** Bracketing vs DOL.
- ✅ **Merging:** Combining L/S frames linearly.
- ✅ **Tone Mapping:** Compressing range for display.
- ✅ **Artifacts:** Ghosting and Banding.

**Next:** Day 33 - Flash & Strobe Control.

---

**Day 32 Complete** | Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control
