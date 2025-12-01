# Day 35: Image Quality (IQ) Tuning & Evaluation
## Phase 3: Camera Systems & ISP | Week 6: Image Quality & Tuning

---

## 🎯 Learning Objectives
1.  **Understand** the difference between Objective (Metrics) and Subjective (Preference) Image Quality.
2.  **Measure** Key Performance Indicators (KPIs): Sharpness (MTF), Noise (SNR), Color Accuracy (DeltaE), Dynamic Range.
3.  **Perform** Subjective Tuning for specific use cases (Portrait, Landscape, Low Light).
4.  **Use** IQ Tuning Tools (Imatest, IQStudio).
5.  **Debug** common IQ artifacts (Moire, Purple Fringing, Texture Loss).
6.  **Create** a Tuning Roadmap for a new sensor integration.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera, Test Charts (ISO 12233, Macbeth, Grey Card, Dead Leaves).
*   **Software:** Imatest (Trial) or Open Source alternatives (MTF Mapper), Python.
*   **Knowledge:** Fourier Transform (MTF), Colorimetry.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Objective Metrics (The Science)
*   **Sharpness (MTF50):** The spatial frequency where contrast drops to 50%. Measured in Line Pairs per Picture Height (LP/PH) or Cycles/Pixel.
*   **Noise (SNR):** Signal-to-Noise Ratio. Measured in dB.
*   **Dynamic Range (DR):** The ratio between the brightest saturating signal and the noise floor. Measured in dB or Stops.
*   **Color Accuracy (DeltaE):** The perceptual distance between the captured color and the reference color. $\Delta E < 2.3$ is "Just Noticeable Difference" (JND).
*   **Texture Preservation:** Measured using "Dead Leaves" chart to see how well random textures are preserved vs noise reduction.

### 🔹 Part 2: Subjective Tuning (The Art)
*   **Memory Colors:** Sky Blue, Grass Green, Skin Tone. Humans prefer these to be "pleasing" rather than "accurate" (e.g., slightly more saturated blue, slightly warmer skin).
*   **Preference:**
    *   **Asian Market:** Prefers brighter skin, less noise, softer texture.
    *   **Western Market:** Prefers more contrast, more texture, tolerates grain.
*   **Use Cases:**
    *   **Portrait:** Soften skin (reduce local contrast), keep eyes sharp.
    *   **Landscape:** High saturation, high sharpness.

### 🔹 Part 3: The Tuning Process
1.  **Bring-up:** Get an image. Fix Black Level.
2.  **Linear Tuning:** LSC, AWB, CCM (Objective).
3.  **Non-Linear Tuning:** Gamma, Tone Mapping (Subjective).
4.  **Detail Tuning:** NR vs Sharpening (The hardest part).
5.  **Field Testing:** Real-world scenarios.

---

## 💻 Implementation Examples

### Example 1: Calculating SNR (Python)

```python
import cv2
import numpy as np

def calculate_snr(image_path, roi):
    """
    Calculate Signal-to-Noise Ratio (dB) from a Grey Card ROI.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x, y, w, h = roi
    patch = img[y:y+h, x:x+w]
    
    mean_val = np.mean(patch)
    std_dev = np.std(patch)
    
    if std_dev == 0:
        return float('inf')
        
    snr_linear = mean_val / std_dev
    snr_db = 20 * np.log10(snr_linear)
    
    return snr_db, mean_val, std_dev

# Usage
# snr, mean, std = calculate_snr("grey_card_iso100.png", (100, 100, 50, 50))
# print(f"SNR: {snr:.2f} dB")
```

### Example 2: Calculating MTF (Slanted Edge) - Concept

1.  **Extract Edge:** Find the ROI with a slanted black/white edge.
2.  **ESF (Edge Spread Function):** Average the scanlines to get a super-sampled edge profile.
3.  **LSF (Line Spread Function):** Derivative of ESF.
4.  **MTF (Modulation Transfer Function):** FFT of LSF.
5.  **MTF50:** Find frequency where MTF = 0.5.

```python
def simple_edge_sharpness(image_path):
    """
    Simplified sharpness metric (Gradient Magnitude).
    Not true MTF, but good for relative comparison.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Sobel Gradient
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    mag = np.sqrt(gx**2 + gy**2)
    
    # Average of top 10% strongest edges
    flat_mag = np.sort(mag.flatten())
    top_10_percent = flat_mag[int(len(flat_mag)*0.9):]
    
    return np.mean(top_10_percent)
```

### Example 3: Color Difference (DeltaE 76)

```python
def calculate_delta_e(rgb_measured, rgb_ref):
    """
    Calculate DeltaE (CIE76) between two RGB colors.
    Requires RGB -> Lab conversion.
    """
    # Convert to Lab (using OpenCV)
    c1 = np.uint8([[rgb_measured]])
    c2 = np.uint8([[rgb_ref]])
    
    lab1 = cv2.cvtColor(c1, cv2.COLOR_RGB2Lab)
    lab2 = cv2.cvtColor(c2, cv2.COLOR_RGB2Lab)
    
    L1, a1, b1 = lab1[0][0]
    L2, a2, b2 = lab2[0][0]
    
    # Euclidean Distance in Lab space
    delta_e = np.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)
    
    return delta_e
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Golden Sample" Calibration

**Objective:** Tune CCM for a new sensor.

**Steps:**
1.  Capture Macbeth chart under D65 light.
2.  Extract RGB values.
3.  Run optimization (Day 24 Lab).
4.  **Validation:** Capture the chart again with the new CCM.
5.  **Check:** Are the grey patches neutral (R=G=B)? Are the colors accurate?

### Lab 2: Noise vs Detail Trade-off

**Objective:** Tune NR strength.

**Steps:**
1.  Capture a "Dead Leaves" chart (or a carpet/fabric) at ISO 800.
2.  Apply NR with Strength 0, 50, 100.
3.  **Observation:**
    *   0: Noisy, sharp texture.
    *   50: Less noise, texture preserved.
    *   100: No noise, texture blurred ("plastic").
4.  **Decision:** Pick the setting where noise is "acceptable" but texture is still visible.

### Lab 3: Purple Fringing (Chromatic Aberration)

**Objective:** Identify lens issues.

**Steps:**
1.  Capture tree branches against a bright white sky.
2.  Zoom in on the edges.
3.  **Observation:** You might see purple or green fringes.
4.  **Fix:** This is an optical flaw. Can be corrected in ISP (CAC - Chromatic Aberration Correction) by warping R/B channels to match G.

---

## 🐛 Debugging Techniques

### Debug 1: Moire Patterns

**Symptom:** Strange rainbow swirls on striped shirts or brick walls.

**Cause:**
*   Detail frequency > Nyquist frequency of sensor.
*   Demosaicing failure.
*   **Fix:** Use an OLPF (Optical Low Pass Filter) on the lens. Or stronger Chroma Denoising.

### Debug 2: Shading (Vignetting)

**Symptom:** Corners are dark or colored.

**Cause:**
*   LSC not tuned for the specific lens.
*   LSC tuned for Daylight used under Tungsten (Spectral sensitivity change).
*   **Fix:** Re-calibrate LSC. Use Dual-LSC.

---

## ⚡ Performance Optimization

### Optimization 1: Tuning for Bitrate

*   High noise kills video encoders (H.264). The encoder spends bits encoding random noise.
*   **Strategy:** Stronger Temporal NR is preferred for video to reduce bitrate, even if it softens the image slightly.

### Optimization 2: Region of Interest (ROI) Tuning

*   Face Detection can drive tuning.
*   If a face is detected, reduce sharpening on the face ROI (to hide wrinkles) but keep background sharp.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is "Sharpness" subjective?**
2.  **What is the difference between SNR 10 and SNR 40?**
3.  **Why do we use the "Lab" color space for DeltaE?**
4.  **How does "Lens Flare" affect Dynamic Range measurements?**

### Practical Challenges

1.  **Create a "Tuning Report" template:** A document listing all KPIs (MTF, SNR, etc.) for a camera release sign-off.
2.  **Design a subjective test plan:** "Take 50 photos: 10 Portrait, 10 Landscape, 10 Low Light, 10 Macro, 10 HDR."

---

## 📚 Further Reading & Resources

### Standards
*   **ISO 12233:** Resolution measurement.
*   **ISO 15739:** Noise measurement.

### Tools
*   **Imatest:** The bible of IQ testing.
*   **DXOMARK:** Study their testing protocols.

---

## 🎓 Summary

Today we covered:
- ✅ **Objective Metrics:** MTF, SNR, DeltaE.
- ✅ **Subjective Tuning:** The "Look" of the camera.
- ✅ **Trade-offs:** Noise vs Detail.
- ✅ **Artifacts:** Moire, Fringing.
- ✅ **Process:** From bring-up to sign-off.

**Next:** Day 36 - Camera Driver Development (V4L2 Deep Dive).

---

**Day 35 Complete** | Phase 3: Camera Systems & ISP | Week 6: Image Quality & Tuning
