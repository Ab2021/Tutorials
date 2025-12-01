# Day 22: ISP Architecture & Raw Processing
## Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development

---

## 🎯 Learning Objectives
1. **Design** a modular Software ISP (Image Signal Processor) architecture
2. **Implement** Black Level Correction (BLC) with calibration data
3. **Develop** Lens Shading Correction (LSC) using mesh grids
4. **Create** Defect Pixel Correction (DPC) algorithms (Static vs Dynamic)
5. **Analyze** Raw image statistics (Histogram, Noise Profile)
6. **Debug** raw domain artifacts (vignetting, color cast, hot pixels)

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Raw Camera Capture (e.g., Raspberry Pi HQ Camera, or saved .raw files)
*   **Software:** C/C++ Compiler, OpenCV (for visualization only), Python (for prototyping)
*   **Data:** Raw Bayer images (RGGB, BGGR) with metadata (bit depth, dimensions)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ISP Pipeline Architecture

An ISP converts raw sensor data (Bayer) into a viewable image (RGB/YUV). It can be implemented in Hardware (FPGA/ASIC) or Software (CPU/GPU).

**Typical Pipeline Stages:**
1.  **Raw Domain:** Processing on Bayer data. Linear, scene-referred.
    *   Black Level Correction (BLC)
    *   Lens Shading Correction (LSC)
    *   Defect Pixel Correction (DPC)
    *   Raw Noise Reduction (RawNR)
    *   White Balance Gain (AWB Gain)
2.  **Bayer Domain:**
    *   Demosaicing (Debayer)
3.  **RGB Domain:** Linear or Gamma-corrected.
    *   Color Correction Matrix (CCM)
    *   Gamma Correction
    *   Tone Mapping (HDR)
    *   3D LUT
4.  **YUV Domain:**
    *   Color Space Conversion (CSC)
    *   Luma/Chroma Noise Reduction
    *   Sharpening

### 🔹 Part 2: Black Level Correction (BLC)

Sensors have a "dark current" even with no light. Pixel values are offset from 0.
*   **Pedestal:** Sensors typically output a pedestal (e.g., 64 for 10-bit) to preserve noise floor.
*   **Correction:** Subtract this value. `Pixel_Out = max(0, Pixel_In - Black_Level)`
*   **Channels:** Black level might differ for R, Gr, Gb, B channels.

### 🔹 Part 3: Lens Shading Correction (LSC)

Lenses are brighter in the center and darker at the corners (Vignetting).
*   **Cos^4 Law:** Natural falloff.
*   **Color Shading:** Different wavelengths refract differently, causing color shifts at corners.
*   **Correction:** Multiply pixels by a gain map (Mesh Shading).
    *   `Pixel_Out(x,y) = Pixel_In(x,y) * Gain(x,y)`

### 🔹 Part 4: Defect Pixel Correction (DPC)

Sensors have "Hot" (always bright) and "Dead" (always dark) pixels.
*   **Static DPC:** Uses a factory calibration map of known bad pixels.
*   **Dynamic DPC:** Detects outliers on-the-fly by comparing a pixel to its neighbors.

---

## 💻 Implementation Examples

### Example 1: Software ISP Framework (C++)

We'll define a modular structure for our ISP.

```cpp
/**
 * @file simple_isp.h
 * @brief Basic Software ISP Structure
 */

#include <vector>
#include <cstdint>
#include <string>

struct RawImage {
    uint16_t* data;       // 16-bit container for 10/12/14-bit raw
    int width;
    int height;
    int bit_depth;
    std::string bayer_pattern; // "RGGB", "BGGR", etc.
};

struct ISPParams {
    // BLC
    uint16_t black_level[4]; // R, Gr, Gb, B
    
    // LSC
    std::vector<float> lsc_mesh;
    int lsc_mesh_width;
    int lsc_mesh_height;
    
    // DPC
    float dpc_threshold;
};

class SimpleISP {
public:
    SimpleISP(int width, int height, int bit_depth);
    ~SimpleISP();
    
    void process(RawImage& input, RawImage& output, const ISPParams& params);

private:
    void apply_blc(RawImage& img, const uint16_t* black_level);
    void apply_lsc(RawImage& img, const std::vector<float>& mesh, int mw, int mh);
    void apply_dpc(RawImage& img, float threshold);
    
    int width_;
    int height_;
};
```

### Example 2: Black Level Correction Implementation

```cpp
/**
 * @brief Apply Black Level Correction
 */
void SimpleISP::apply_blc(RawImage& img, const uint16_t* black_level) {
    // Bayer pattern assumption: RGGB
    // Row 0: R G R G ...
    // Row 1: G B G B ...
    
    #pragma omp parallel for
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int idx = y * img.width + x;
            uint16_t val = img.data[idx];
            uint16_t bl = 0;
            
            // Determine channel based on position (RGGB)
            if (y % 2 == 0) {
                if (x % 2 == 0) bl = black_level[0]; // R
                else            bl = black_level[1]; // Gr
            } else {
                if (x % 2 == 0) bl = black_level[2]; // Gb
                else            bl = black_level[3]; // B
            }
            
            // Subtract and clamp
            if (val > bl) {
                img.data[idx] = val - bl;
            } else {
                img.data[idx] = 0;
            }
        }
    }
}
```

### Example 3: Lens Shading Correction (Mesh Based)

Using Bilinear Interpolation to sample the gain mesh.

```cpp
/**
 * @brief Apply Lens Shading Correction
 */
void SimpleISP::apply_lsc(RawImage& img, const std::vector<float>& mesh, int mw, int mh) {
    // Mesh covers the image. We need to interpolate gain for each pixel.
    float cell_w = (float)img.width / (mw - 1);
    float cell_h = (float)img.height / (mh - 1);
    
    #pragma omp parallel for
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            // Find mesh cell coordinates
            float gx = x / cell_w;
            float gy = y / cell_h;
            
            int ix = (int)gx;
            int iy = (int)gy;
            
            // Clamp to mesh bounds
            if (ix >= mw - 1) ix = mw - 2;
            if (iy >= mh - 1) iy = mh - 2;
            
            // Fractional part
            float fx = gx - ix;
            float fy = gy - iy;
            
            // Bilinear interpolation of gain
            // Note: In real ISP, we have 4 meshes (one per channel)
            // Simplified here to use one mesh
            
            float g00 = mesh[iy * mw + ix];
            float g10 = mesh[iy * mw + (ix + 1)];
            float g01 = mesh[(iy + 1) * mw + ix];
            float g11 = mesh[(iy + 1) * mw + (ix + 1)];
            
            float gain = (1 - fx) * (1 - fy) * g00 +
                         fx * (1 - fy) * g10 +
                         (1 - fx) * fy * g01 +
                         fx * fy * g11;
            
            // Apply gain
            float val = (float)img.data[y * img.width + x] * gain;
            
            // Clamp to max value (e.g., 1023 for 10-bit)
            int max_val = (1 << img.bit_depth) - 1;
            if (val > max_val) val = max_val;
            
            img.data[y * img.width + x] = (uint16_t)val;
        }
    }
}
```

### Example 4: Dynamic Defect Pixel Correction

Simple 3x3 median-like filter to detect spikes.

```cpp
/**
 * @brief Apply Dynamic DPC
 */
void SimpleISP::apply_dpc(RawImage& img, float threshold) {
    // Skip borders
    #pragma omp parallel for
    for (int y = 2; y < img.height - 2; y++) {
        for (int x = 2; x < img.width - 2; x++) {
            int idx = y * img.width + x;
            uint16_t center = img.data[idx];
            
            // Check neighbors of SAME color
            // For RGGB, neighbors are at stride 2
            uint16_t neighbors[8];
            neighbors[0] = img.data[(y-2)*img.width + (x-2)];
            neighbors[1] = img.data[(y-2)*img.width + x];
            neighbors[2] = img.data[(y-2)*img.width + (x+2)];
            neighbors[3] = img.data[y*img.width + (x-2)];
            neighbors[4] = img.data[y*img.width + (x+2)];
            neighbors[5] = img.data[(y+2)*img.width + (x-2)];
            neighbors[6] = img.data[(y+2)*img.width + x];
            neighbors[7] = img.data[(y+2)*img.width + (x+2)];
            
            // Calculate average of neighbors
            float sum = 0;
            for(int i=0; i<8; i++) sum += neighbors[i];
            float avg = sum / 8.0f;
            
            // If center is significantly different, replace it
            if (abs(center - avg) > threshold) {
                // Simple replacement with average (or median)
                img.data[idx] = (uint16_t)avg;
            }
        }
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Black Level Calibration

**Objective:** Determine the Black Level of your camera.

**Steps:**
1.  Cover the lens completely (Lens Cap ON).
2.  Capture a raw frame in a dark room.
3.  Calculate the mean value of the image.
4.  **Result:** This mean value is your Black Level (e.g., 64).
5.  **Advanced:** Calculate mean for each Bayer channel (R, Gr, Gb, B) separately. They might differ!

### Lab 2: LSC Mesh Generation

**Objective:** Create a Lens Shading Correction mesh.

**Steps:**
1.  Point camera at a uniform white surface (Flat Field).
2.  Ensure uniform lighting (no shadows).
3.  Capture a raw frame.
4.  **Processing:**
    *   Find the maximum pixel value (usually in center).
    *   For every pixel, `Gain(x,y) = Max_Value / Pixel(x,y)`.
    *   Downsample this gain map to a mesh (e.g., 16x12).
5.  **Verification:** Apply this mesh to the flat field image. It should become perfectly flat (uniform brightness).

### Lab 3: Hot Pixel Detection

**Objective:** Find hot pixels.

**Steps:**
1.  Capture a dark frame with high analog gain and long exposure.
2.  Threshold the image (e.g., > 1000).
3.  Any pixel above threshold is a "Hot Pixel".
4.  Create a list of (x, y) coordinates for Static DPC.

---

## 🐛 Debugging Techniques

### Debug 1: Color Cast in Shadows

**Symptom:** Dark areas look purple or green.

**Cause:** Incorrect Black Level subtraction.
*   If you subtract too much Green, shadows look Purple.
*   If you subtract too much Red/Blue, shadows look Green.

**Fix:** Re-calibrate Black Level for each channel precisely.

### Debug 2: "Donut" Effect in LSC

**Symptom:** Image is bright in center, dark in middle, bright at corners (or inverted).

**Cause:** LSC Mesh does not match the lens/sensor combination.
*   Wrong mesh applied.
*   Mesh applied to cropped image without coordinate offset.

**Fix:** Ensure LSC mesh coordinates align with the active pixel array.

---

## ⚡ Performance Optimization

### Optimization 1: Fixed Point Arithmetic

*   Floating point (float) is slow on some embedded ISPs/DSPs.
*   **Technique:** Use Q-format (e.g., Q4.12 for gains).
    *   Gain of 1.5 -> `1.5 * 4096 = 6144`.
    *   `Pixel * Gain` -> `(Pixel * 6144) >> 12`.

### Optimization 2: Vectorization (SIMD)

*   Process 8 or 16 pixels at a time using NEON (ARM) or AVX (x86).
*   BLC and LSC are perfect candidates for SIMD as operations are independent per pixel.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do we perform BLC *before* LSC?**
2.  **What happens if you apply LSC to a saturated pixel?**
3.  **Why do we check neighbors of the *same* color in DPC?**
4.  **How does bit-depth affect the Black Level value?**

### Practical Challenges

1.  **Implement a "Vignetting Model"** that generates a synthetic LSC mesh based on `Gain = 1 + k * r^2`.
2.  **Optimize the LSC implementation** to use integer math only (no floats).

---

## 📚 Further Reading & Resources

### Books
*   **"Digital Image Processing Pipeline"** - A generic term, look for sensor datasheets.
*   **"Camera Image Processing"** by Rastislav Lukac.

### Open Source
*   **libcamera:** Study the `ipa` (Image Processing Algorithms) modules.
*   **Raspberry Pi ISP:** Documentation on the Broadcom ISP pipeline.

---

## 🎓 Summary

Today we covered:
- ✅ **ISP Pipeline:** The journey from Raw to RGB.
- ✅ **BLC:** Removing the dark current pedestal.
- ✅ **LSC:** Correcting lens vignetting using gain meshes.
- ✅ **DPC:** Fixing hot/dead pixels.
- ✅ **Implementation:** Building a C++ framework for Software ISP.

**Next:** Day 23 - Demosaicing & Interpolation Algorithms.

---

**Day 22 Complete** | Phase 3: Camera Systems & ISP | Week 4: ISP Pipeline Development
