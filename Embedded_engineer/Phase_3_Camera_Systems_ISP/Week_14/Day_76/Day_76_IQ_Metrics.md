# Day 76: Image Quality Metrics (Sharpness, Noise, Dynamic Range)
## Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Define** Objective Image Quality (IQ) metrics vs Subjective preference.
2.  **Measure** Sharpness using MTF (Modulation Transfer Function) and SFR (Spatial Frequency Response).
3.  **Quantify** Noise using SNR (Signal-to-Noise Ratio) and Visual Noise.
4.  **Calculate** Dynamic Range (DR) in dB using OECF charts.
5.  **Analyze** Distortion (TV Distortion, SMIA TV Distortion).
6.  **Use** Open-source tools (e.g., MTF Mapper) to validate lens performance.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera, Tripod, Lighting (D50/D65 High CRI).
*   **Targets:** ISO 12233 (Resolution), OECF (Dynamic Range), Macbeth ColorChecker.
*   **Software:** MTF Mapper (Open Source) or Imatest (Trial).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Sharpness & Resolution (MTF)
*   **Resolution:** The count of pixels (e.g., 1920x1080).
*   **Sharpness:** The contrast at boundaries. A 4K image can be blurry (low sharpness).
*   **MTF (Modulation Transfer Function):** The standard metric.
    *   **MTF50:** The spatial frequency (Line Pairs per Picture Height - LP/PH) where contrast drops to 50%.
    *   **Slanted Edge Method:** We photograph a tilted black edge. The algorithm analyzes the transition (Edge Spread Function -> Line Spread Function -> FFT -> MTF).

### 🔹 Part 2: Noise (SNR)
*   **Photon Shot Noise:** Randomness of light ($\sqrt{Signal}$). Dominates in bright light.
*   **Read Noise:** Electronic noise. Dominates in low light.
*   **SNR (dB):** $20 \log_{10}(\frac{Mean Signal}{Standard Deviation})$.
*   **Target:** For automotive, we want SNR > 30dB at 1 Lux (very hard!).

### 🔹 Part 3: Dynamic Range (DR)
*   The ratio between the brightest signal (saturation) and the darkest signal (noise floor).
*   **Formula:** $DR = 20 \log_{10}(\frac{Full Well Capacity}{Read Noise})$.
*   **HDR Sensors:** Use multiple exposures (Long, Short) to extend DR to 120dB or 140dB.

---

## 💻 Implementation Examples

### Example 1: Calculating SNR (Python)

```python
import cv2
import numpy as np

def calculate_snr(image_path, roi):
    # Load image (Grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Crop to a uniform gray patch (e.g., Macbeth Patch 22)
    x, y, w, h = roi
    patch = img[y:y+h, x:x+w]
    
    # Calculate Statistics
    mean_signal = np.mean(patch)
    std_dev = np.std(patch)
    
    # Calculate SNR
    if std_dev == 0:
        return float('inf')
        
    snr_linear = mean_signal / std_dev
    snr_db = 20 * np.log10(snr_linear)
    
    return snr_db, mean_signal

# Usage
snr, signal = calculate_snr("test_chart.jpg", (500, 500, 100, 100))
print(f"Signal: {signal:.2f}, SNR: {snr:.2f} dB")
```

### Example 2: Slanted Edge MTF (Concept)

1.  **Extract Edge:** Find the edge in the ROI.
2.  **Super-sampling:** Project pixel values onto the edge normal vector to create a high-resolution edge profile (ESF).
3.  **Differentiation:** $LSF = d(ESF)/dx$.
4.  **FFT:** $MTF = |FFT(LSF)|$.

*Note: Implementing this from scratch is complex. Use `mtf_mapper` CLI.*

```bash
# Using MTF Mapper (CLI)
mtf_mapper -a -s -f test_chart.jpg output_dir
# Generates annotated images with MTF50 values on edges.
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Measuring Lens Sharpness

**Objective:** Is the lens focused? Is it decentered?

**Steps:**
1.  Print an ISO 12233 Chart (or use a monitor).
2.  Capture an image.
3.  Run `mtf_mapper`.
4.  **Analyze:** Check MTF50 in the Center vs Corners.
5.  **Pass Criteria:** Center > 0.4 cycles/pixel. Corners > 0.2 cycles/pixel.
6.  **Fail:** If one corner is blurry and others are sharp -> **Decentering** (Tilt).

### Lab 2: Noise vs Gain

**Objective:** Plot the Noise Curve.

**Steps:**
1.  Point camera at a gray card.
2.  Capture images at Gain 1x, 2x, 4x, 8x, 16x.
3.  Calculate SNR for each (Example 1).
4.  **Plot:** SNR (y-axis) vs Gain (x-axis).
5.  **Observation:** SNR drops by 6dB for every 2x Gain (roughly).

### Lab 3: Dynamic Range Test

**Objective:** Verify HDR.

**Steps:**
1.  Set up a high-contrast scene (Dark tunnel + Bright exit).
2.  Capture with Linear Mode (Single Exposure). Result: Blown out exit or black tunnel.
3.  Capture with HDR Mode (WDR).
4.  **Observation:** Both areas should have details.
5.  **Measurement:** Use an OECF chart (transmissive with backlight) to measure the exact steps visible.

---

## 🐛 Debugging IQ Issues

### Debug 1: "Soft" Images

**Symptom:** MTF is low everywhere.

**Cause:**
*   Focus is wrong (Back focus / Front focus).
*   Lens resolution is too low for the sensor (e.g., 2MP lens on 8MP sensor).
*   Motion blur during capture.
*   **Fix:** Adjust focus. Check lens spec (LP/mm).

### Debug 2: Fixed Pattern Noise (FPN)

**Symptom:** Vertical lines visible in low light.

**Cause:**
*   Column Parallel ADC variations.
*   **Fix:** DPC (Defect Pixel Correction) or FPN Correction in ISP (subtract a dark frame).

---

## ⚡ Performance Optimization

### Optimization 1: Tuning Sharpening (USM)

*   ISP Sharpening (Unsharp Mask) artificially boosts MTF.
*   **Risk:** "Halo" artifacts on edges.
*   **Tuning:** Increase sharpening strength until halos appear, then back off.

### Optimization 2: Tuning NR (Noise Reduction)

*   Spatial NR blurs textures.
*   **Trade-off:** Less Noise vs More Detail (Texture Preservation).
*   **Metric:** "Texture Acutance" (measuring MTF on low-contrast textures like "Dead Leaves" chart).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is MTF50 preferred over "Limiting Resolution"?** (MTF50 correlates better with perceived sharpness. Limiting resolution just tells you the vanishing point).
2.  **What is the "Nyquist Frequency"?** (0.5 cycles/pixel. The maximum frequency a sensor can resolve).
3.  **Why do we measure SNR on a gray patch, not a black one?** (Black is dominated by Read Noise. Gray includes Shot Noise).
4.  **What is "Veiling Glare"?** (Stray light reducing contrast).

### Practical Challenges

1.  **Build a Low-Cost Test Lab:** Use a high-CRI LED bulb, a printed chart, and a dark room. Calibrate the light uniformity using a Lux meter app.
2.  **Automate MTF Check:** Write a Python script that calls `mtf_mapper`, parses the output CSV, and returns "PASS/FAIL" based on a threshold.

---

## 📚 Further Reading & Resources

### Standards
*   **ISO 12233:** Resolution measurement.
*   **EMVA 1288:** Standard for machine vision sensor characterization.

### Tools
*   **MTF Mapper (SourceForge).**

---

## 🎓 Summary

Today we covered:
- ✅ **Sharpness:** MTF, Slanted Edge.
- ✅ **Noise:** SNR, Shot Noise vs Read Noise.
- ✅ **DR:** Decibels, OECF.
- ✅ **Tools:** Python & MTF Mapper.
- ✅ **Tuning:** The balance of Sharpness vs Artifacts.

**Next:** Day 77 - Objective Testing with Imatest & Open Source Tools.

---

**Day 76 Complete** | Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance
