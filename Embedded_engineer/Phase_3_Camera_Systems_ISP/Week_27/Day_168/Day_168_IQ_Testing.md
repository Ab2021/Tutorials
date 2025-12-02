# Day 168: Image Quality Testing (Imatest & Metrics)
## Phase 3: Camera Systems & ISP | Week 27: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Define** Objective Image Quality (IQ) metrics: Sharpness (MTF), Noise (SNR), Dynamic Range (DR), Color Accuracy ($\Delta E$).
2.  **Use** Imatest (or Open Source equivalents) to analyze test charts.
3.  **Measure** MTF50 using a Slanted Edge Chart (ISO 12233).
4.  **Analyze** Color Accuracy using a Macbeth ColorChecker.
5.  **Automate** IQ testing using Python.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera, Tripod, Uniform Lighting (D65).
*   **Charts:** ISO 12233 Resolution Chart, Macbeth ColorChecker (24 patches).
*   **Software:** Imatest (Trial) or `colour-science` / `scikit-image` (Python).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Sharpness (MTF)
*   **MTF (Modulation Transfer Function):** How well the lens/sensor preserves contrast at different spatial frequencies.
*   **MTF50:** The frequency (Line Pairs per Picture Height - LP/PH) where contrast drops to 50%.
*   **Slanted Edge Method:** We photograph a tilted black/white edge. The software analyzes the transition (Edge Spread Function -> Line Spread Function -> FFT -> MTF).

### 🔹 Part 2: Noise (SNR)
*   **SNR (Signal-to-Noise Ratio):** $20 \log_{10}(\frac{Signal}{Noise})$.
*   **Measurement:** Photograph a gray patch.
    *   Signal = Mean pixel value ($\mu$).
    *   Noise = Standard Deviation ($\sigma$).
    *   $SNR = \mu / \sigma$.

### 🔹 Part 3: Dynamic Range
*   The ratio between the brightest saturating signal and the noise floor.
*   Measured in dB or Stops.
*   **Chart:** OECF (Opto-Electronic Conversion Function) chart with grayscale patches of increasing density.

---

## 💻 Implementation Examples

### Example 1: Calculating MTF50 (Python)

Simplified Slanted Edge analysis.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft

def calculate_mtf(edge_image):
    # 1. Get Edge Profile (Average across rows)
    # Assume edge is roughly vertical
    profile = np.mean(edge_image, axis=0)
    
    # 2. Derivative (Line Spread Function - LSF)
    lsf = np.abs(np.gradient(profile))
    
    # 3. FFT (Modulation Transfer Function - MTF)
    mtf = np.abs(fft(lsf))
    mtf = mtf[:len(mtf)//2] # Positive frequencies
    mtf = mtf / mtf[0] # Normalize to 1.0 at DC
    
    # 4. Find MTF50
    freqs = np.linspace(0, 0.5, len(mtf)) # Cycles per pixel
    mtf50_idx = np.where(mtf < 0.5)[0][0]
    mtf50 = freqs[mtf50_idx]
    
    return freqs, mtf, mtf50

# Load ROI of a slanted edge
img = cv2.imread("slanted_edge_roi.png", 0)
freqs, mtf, val = calculate_mtf(img)

print(f"MTF50: {val:.3f} cycles/pixel")
plt.plot(freqs, mtf)
plt.axhline(0.5, color='r')
plt.show()
```

### Example 2: Color Accuracy ($\Delta E$)

Using `colour-science`.

```python
import colour
import numpy as np

# Measured RGB (from camera)
RGB_measured = np.array([0.5, 0.4, 0.3])

# Reference RGB (from chart spec, e.g., sRGB)
RGB_ref = np.array([0.55, 0.42, 0.31])

# Convert to Lab
Lab_measured = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(RGB_measured))
Lab_ref = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(RGB_ref))

# Calculate Delta E (CIE 2000)
dE = colour.delta_E(Lab_measured, Lab_ref, method="CIE 2000")

print(f"Delta E: {dE:.2f}")
# < 2.0 is Good. > 5.0 is Bad.
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Shoot the Chart

**Objective:** Capture valid test data.

**Steps:**
1.  Setup the ISO 12233 Chart.
2.  Align camera center. Ensure even lighting (no glare).
3.  Capture RAW and ISP-Processed (JPG) images.
4.  **Observation:** JPG looks sharper due to ISP Sharpening, but might have "Halos" (Overshoot).

### Lab 2: Measure SNR vs ISO

**Objective:** Characterize the sensor.

**Steps:**
1.  Set Gain/ISO to Min (e.g., 100). Photograph a gray card. Measure SNR.
2.  Set Gain/ISO to Max (e.g., 6400). Photograph same card. Measure SNR.
3.  **Result:** SNR drops drastically at high gain.
4.  **Plot:** SNR (dB) vs Gain (dB).

### Lab 3: Lens Shading (Vignetting)

**Objective:** Measure Fall-off.

**Steps:**
1.  Photograph a uniform white wall (Flat Field).
2.  Measure brightness at Center vs Corners.
3.  **Formula:** $Shading \% = \frac{Corner}{Center} \times 100$.
4.  **Goal:** > 80% is good. < 50% is noticeable darkening.

---

## 🐛 Debugging Image Quality

### Debug 1: "Soft Images"

**Symptom:** MTF50 is low (e.g., 0.1 cycles/pixel).

**Cause:**
*   **Focus:** Lens is out of focus.
*   **Motion Blur:** Shutter speed too slow.
*   **Lens Quality:** Cheap plastic lens.
*   **Fix:** Adjust focus. Use a tripod. Stop down aperture (increase F-number) to improve sharpness (up to diffraction limit).

### Debug 2: "Purple Fringing"

**Symptom:** Purple edges around high-contrast objects.

**Cause:**
*   **Chromatic Aberration:** Different wavelengths focus at different points.
*   **Fix:** Better lens (Achromatic doublet). Or Software Correction (CAC) in ISP.

---

## ⚡ Performance Optimization

### Optimization 1: Automated Testing

*   Don't measure manually.
*   Write a script that captures an image, detects the chart (using ArUco markers on corners), extracts patches, and computes metrics automatically.
*   Run this on every firmware build (CI/CD).

### Optimization 2: Golden Sample

*   Keep one "Golden" camera unit that is perfectly calibrated.
*   Compare every production unit against the Golden unit.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why use a "Slanted" edge instead of a vertical one?** (To achieve sub-pixel resolution. The edge cuts across pixels at different phases, allowing us to reconstruct the LSF with higher precision).
2.  **What is "Nyquist Frequency"?** (0.5 cycles/pixel. The maximum frequency that can be resolved. Frequencies above this cause Aliasing/Moiré).
3.  **Why is $\Delta E$ calculated in Lab space, not RGB?** (Lab is perceptually uniform. A distance of 1.0 in Lab looks like the same color difference to the human eye everywhere in the space).

### Practical Challenges

1.  **Build a Light Box:** Use a cardboard box and LED strips to create a controlled lighting environment for testing.
2.  **Measure Distortion:** Photograph a grid. Measure the curvature of the lines.

---

## 📚 Further Reading & Resources

### Documentation
*   **Imatest Documentation (The Bible of IQ).**
*   **ISO 12233 Standard.**

---

## 🎓 Summary

Today we covered:
- ✅ **MTF:** Sharpness measurement.
- ✅ **SNR:** Noise measurement.
- ✅ **Charts:** ISO 12233 & Macbeth.
- ✅ **Delta E:** Color error.
- ✅ **Automation:** Python analysis.

**Next:** Day 169 - Automotive Standards (AEC-Q100).

---

**Day 168 Complete** | Phase 3: Camera Systems & ISP | Week 27: Testing, Validation & Compliance


