# Day 57: Production Testing & Calibration
## Phase 3: Camera Systems & ISP | Week 10: Manufacturing, Calibration & Tuning

---

## 🎯 Learning Objectives
1.  **Understand** the difference between Lab Calibration (Golden Sample) and Production Calibration (Every Unit).
2.  **Perform** Intrinsic Calibration (Lens Distortion & Focal Length) on the line.
3.  **Implement** Lens Shading Correction (LSC) calibration per unit.
4.  **Detect** Blemishes (Bad Pixels, Dust) automatically.
5.  **Measure** MTF (Modulation Transfer Function) for Pass/Fail grading.
6.  **Write** calibration data to OTP (One-Time Programmable) memory.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera Module, Uniform Light Source (Integrating Sphere), Test Charts (Checkerboard, ISO 12233).
*   **Software:** OpenCV, Imatest (Reference).
*   **Knowledge:** Vignetting, Distortion Models.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Per-Unit Calibration?
*   **Lens Variation:** Every lens has slightly different focal length and distortion due to molding tolerances.
*   **Sensor Variation:** Every sensor has slightly different sensitivity and bad pixels.
*   **Assembly Variation:** Tilt and shift affect the optical center ($c_x, c_y$).
*   **Solution:** Measure these errors for *each* camera and save the correction parameters in the module's EEPROM/OTP.

### 🔹 Part 2: Calibration Stations
1.  **Infinity Focus:** Adjust lens position for infinity.
2.  **LSC/WB (Light Box):** Capture a flat grey field. Calculate Lens Shading and AWB Golden Ratios.
3.  **Chart Test:** Capture a resolution chart. Measure MTF, Distortion, and Intrinsic Matrix ($K$).
4.  **Blemish:** Detect dark spots (dust) or hot pixels.

### 🔹 Part 3: OTP (One-Time Programmable) Memory
*   Small memory (e.g., 8KB) inside the sensor or a separate EEPROM.
*   **Stores:**
    *   Module ID / Serial Number.
    *   LSC Mesh (Lens Shading Correction).
    *   AWB Golden Ratios (R/G, B/G).
    *   Defect Pixel List (DPC).
*   **Driver:** The OS driver reads this OTP at boot and applies corrections.

---

## 💻 Implementation Examples

### Example 1: Lens Shading Calibration (LSC)

Generating the LSC Mesh from a flat field image.

```python
import cv2
import numpy as np

def generate_lsc_mesh(flat_field_img, grid_size=(16, 16)):
    """
    Calculates gain map to flatten the illumination.
    """
    h, w = flat_field_img.shape[:2]
    # 1. Convert to float
    img = flat_field_img.astype(np.float32)
    
    # 2. Find Max Brightness (Center)
    max_val = np.max(img)
    
    # 3. Calculate Gain Map (Inverse of brightness)
    # Gain = Target / Current
    gain_map = max_val / (img + 1e-6)
    
    # 4. Downsample to Grid (e.g., 17x17 points)
    lsc_mesh = cv2.resize(gain_map, grid_size, interpolation=cv2.INTER_AREA)
    
    return lsc_mesh

# Usage
flat = cv2.imread("flat_field.png", 0)
mesh = generate_lsc_mesh(flat)
# Save 'mesh' to OTP
```

### Example 2: Blemish Detection

Finding dust spots.

```cpp
/**
 * @brief Detect Dust Spots
 */
void detect_blemish(cv::Mat& flat_field) {
    cv::Mat blur, diff, thresh;
    
    // 1. Low-pass filter (Background)
    cv::GaussianBlur(flat_field, blur, cv::Size(51, 51), 0);
    
    // 2. Difference (High-pass)
    // Dust spots are high-frequency dips in brightness
    cv::absdiff(flat_field, blur, diff);
    
    // 3. Threshold
    // If pixel is > 5% darker than local average
    cv::threshold(diff, thresh, 10, 255, cv::THRESH_BINARY);
    
    // 4. Count blobs
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.size() > 3) {
        std::cout << "FAIL: Too many dust spots!" << std::endl;
    }
}
```

### Example 3: Writing to OTP (Simulated)

```c
struct otp_data {
    uint32_t module_id;
    uint16_t awb_r_g; // R/G ratio * 1024
    uint16_t awb_b_g; // B/G ratio * 1024
    uint8_t lsc_grid[16][16];
};

void write_otp(struct i2c_client *client, struct otp_data *data) {
    // 1. Enable OTP Program Mode (High Voltage Pump)
    i2c_write(client, REG_OTP_MODE, 0x01);
    
    // 2. Write Data
    uint8_t *bytes = (uint8_t*)data;
    for (int i=0; i<sizeof(*data); i++) {
        i2c_write(client, REG_OTP_START + i, bytes[i]);
    }
    
    // 3. Disable Program Mode
    i2c_write(client, REG_OTP_MODE, 0x00);
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Flat Field Capture

**Objective:** Capture calibration images.

**Steps:**
1.  Set up a uniform light source (or point camera at a white screen).
2.  Set exposure to reach ~80% saturation (mean value 200/255).
3.  Capture 10 frames and average them (to remove temporal noise).
4.  **Result:** You will see the "Vignetting" fall-off (corners are darker).

### Lab 2: MTF Measurement (Slanted Edge)

**Objective:** Measure sharpness.

**Steps:**
1.  Print an ISO 12233 chart.
2.  Capture an image.
3.  Select a slanted edge ROI.
4.  Compute the Edge Spread Function (ESF) -> Line Spread Function (LSF) -> FFT -> MTF.
5.  **Goal:** MTF50 (frequency where contrast drops to 50%) should be > 0.3 cycles/pixel.

### Lab 3: Bad Pixel Mapping

**Objective:** Find dead pixels.

**Steps:**
1.  Cover the lens (Dark Frame).
2.  Set high analog gain.
3.  Find pixels > Threshold (Hot Pixels).
4.  **Result:** A list of (x, y) coordinates to be corrected by the DPC block.

---

## 🐛 Debugging Calibration Issues

### Debug 1: "Color Shift" after LSC

**Symptom:** Corners turn pink or green.

**Cause:**
*   LSC was calibrated with a light source temperature (e.g., 6500K) different from the current scene (e.g., 2700K).
*   Vignetting is wavelength dependent.
*   **Fix:** Calibrate LSC under 3 illuminants (A, D50, D65) and interpolate based on current AWB estimate.

### Debug 2: "Circles" in Image

**Symptom:** Concentric rings visible in flat areas.

**Cause:**
*   LSC grid resolution too low.
*   Quantization error in LSC gain table (8-bit vs 10-bit).
*   **Fix:** Use higher bit-depth for LSC gains or larger grid.

---

## ⚡ Performance Optimization

### Optimization 1: Parallel Testing

*   In production, testing one camera takes 10 seconds.
*   **Solution:** Use a "Nest" that tests 4 or 8 cameras simultaneously.
*   Reduces cycle time to < 2 seconds per unit.

### Optimization 2: Chart Design

*   Design a chart that has:
    *   Grey patches for Noise/Color.
    *   Slanted edges for MTF.
    *   Dot pattern for Distortion.
*   Allows measuring ALL metrics in a single snapshot.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do we average multiple frames for Flat Field calibration?** (To remove Shot Noise).
2.  **What is "Golden Sample" calibration?**
3.  **Why is OTP memory used instead of Flash?** (Cost and size).
4.  **How does "Slanted Edge" MTF work?**

### Practical Challenges

1.  **Implement "Auto-Exposure" for Test:** Write a script that adjusts exposure time until the mean brightness of the chart is 128 +/- 5.
2.  **Decode OTP:** Read a binary dump of OTP memory and parse the LSC mesh back into a heatmap image.

---

## 📚 Further Reading & Resources

### Standards
*   **EMVA 1288:** Standard for Characterization of Image Sensors and Cameras.

### Tools
*   **Imatest:** The industry standard for camera quality analysis.

---

## 🎓 Summary

Today we covered:
- ✅ **Production Line:** The reality of mass production.
- ✅ **Calibration:** LSC, AWB, Intrinsic.
- ✅ **Testing:** MTF, Blemish.
- ✅ **OTP:** Storing the DNA of each camera.
- ✅ **Yield:** Pass/Fail criteria.

**Next:** Day 58 - ISP Tuning Workflow (The Art of Tuning).

---

**Day 57 Complete** | Phase 3: Camera Systems & ISP | Week 10: Manufacturing, Calibration & Tuning
