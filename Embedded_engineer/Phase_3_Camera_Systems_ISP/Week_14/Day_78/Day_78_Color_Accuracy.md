# Day 78: Color Accuracy & White Balance Verification
## Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Validate** Auto-White Balance (AWB) performance under standard illuminants (D65, TL84, A, H).
2.  **Understand** Correlated Color Temperature (CCT) and Tint (Green/Magenta shift).
3.  **Analyze** "Memory Colors" (Skin Tones, Blue Sky, Green Grass) for perceptual accuracy.
4.  **Measure** Color Shading (Lens Color Cast) and verify LSC tuning.
5.  **Test** Mixed Lighting scenarios (e.g., Window + Indoor Light).
6.  **Debug** "Yellow Skin" or "Purple Shadows" issues.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Light Booth (with D65, A, TL84 sources), Macbeth Chart.
*   **Software:** Python Analysis Scripts (from Day 77).
*   **Knowledge:** Planckian Locus, Gray World Assumption.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Standard Illuminants
*   **D65 (6500K):** Average Noon Daylight. Blue-ish.
*   **D50 (5000K):** Horizon Light. Neutral.
*   **TL84 (4000K):** Store Fluorescent. Green spike.
*   **Illuminant A (2856K):** Tungsten / Incandescent. Very Orange.
*   **Horizon (2300K):** Sunrise/Sunset. Extremely Orange.

### 🔹 Part 2: AWB Evaluation Metrics
*   **Gray Neutrality:** Does a gray object look gray?
    *   Target: $\Delta C < 2$ (Chroma error) on gray patches.
*   **Convergence Speed:** How fast does AWB settle when light changes?
*   **Stability:** Does AWB oscillate (flicker) under constant light?

### 🔹 Part 3: Memory Colors
*   Humans are sensitive to specific colors:
    *   **Skin:** Must not look green (sick) or too purple (sunburn).
    *   **Sky:** Must be a nice deep blue, not cyan.
    *   **Grass:** Must be natural green, not neon.
*   **Tuning:** We often "cheat" by pushing these colors towards preferred zones (Preference vs Accuracy).

---

## 💻 Implementation Examples

### Example 1: CCT Calculation (Python)

Estimating Color Temperature from RGB.

```python
import colour

def calculate_cct(rgb_normalized):
    # Convert RGB to XYZ
    xyz = colour.sRGB_to_XYZ(rgb_normalized)
    
    # Convert XYZ to xy Chromaticity
    xy = colour.XYZ_to_xy(xyz)
    
    # Calculate CCT (McCamy's Formula or Robertson's Method)
    cct = colour.xy_to_CCT(xy, method='McCamy')
    
    return cct

# Example: Neutral Gray under Tungsten might look Orange (High R, Low B)
# If AWB works, the RGB of a gray patch should be roughly equal (0.5, 0.5, 0.5)
# resulting in a CCT of ~6500K (if display is D65 calibrated).
```

### Example 2: Color Shading Measurement

Checking if corners are pink/green.

```python
def measure_color_shading(image):
    # Split Channels
    b, g, r = cv2.split(image)
    
    # Calculate R/G and B/G ratios
    rg_ratio = r / g
    bg_ratio = b / g
    
    # Compare Center vs Corner
    h, w = g.shape
    center_rg = rg_ratio[h//2, w//2]
    corner_rg = rg_ratio[0, 0]
    
    # Ratio Deviation
    deviation = abs(center_rg - corner_rg) / center_rg
    
    if deviation > 0.1: # > 10% difference
        print("FAIL: Significant Color Shading Detected")
    else:
        print("PASS: Color Uniformity OK")
```

### Example 3: Skin Tone Vector Scope

Visualizing skin tones in CbCr plane.

```python
# Skin Tone Line usually lies around 105-120 degrees in CbCr
# Convert to YCbCr
ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# Plot Cb vs Cr for skin patches
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Light Booth Sweep

**Objective:** Validate AWB across range.

**Steps:**
1.  Place Macbeth Chart in Light Booth.
2.  Set Light to **D65**. Capture.
3.  Set Light to **TL84**. Capture.
4.  Set Light to **Illuminant A**. Capture.
5.  **Analyze:** Calculate Mean $\Delta C$ for the 6 gray patches in all 3 images.
6.  **Pass Criteria:** $\Delta C < 3$ for all.

### Lab 2: Mixed Lighting Test

**Objective:** Confuse the AWB.

**Steps:**
1.  Place chart near a window (Daylight, 6500K).
2.  Turn on a warm lamp (Tungsten, 2800K) inside.
3.  Capture image showing both sources.
4.  **Observation:**
    *   Simple AWB will pick an average (e.g., 4500K).
    *   Result: Window looks Blue, Interior looks Orange.
    *   **Advanced AWB:** Uses "Dual Illuminant" estimation or Local AWB to correct regions differently (very hard!).

### Lab 3: Skin Tone Preference

**Objective:** Subjective vs Objective.

**Steps:**
1.  Photograph a person.
2.  Tune AWB to be mathematically perfect (Gray card is gray).
3.  Tune AWB to make skin look "nice" (slightly warmer/pinker).
4.  **Survey:** Ask 5 people which they prefer. Usually, the "nice" one wins.

---

## 🐛 Debugging AWB Issues

### Debug 1: Green Spike (Fluorescent)

**Symptom:** Images look green under office lights.

**Cause:**
*   Fluorescent lights have a strong Green spectral peak.
*   AWB algorithm assumes "Gray World" and fails.
*   **Fix:** Implement "Anti-Flicker" (50/60Hz detection) and specific "Fluorescent Detection" logic in AWB stats.

### Debug 2: Pink Corners

**Symptom:** Center is neutral, corners are pink.

**Cause:**
*   Lens Shading (LSC) is correcting for Luma but not Chroma.
*   IR Cut Filter angle dependence.
*   **Fix:** Tune the **Color Shading Correction (CSC)** mesh table.

---

## ⚡ Performance Optimization

### Optimization 1: Temporal Hysteresis

*   Don't change AWB gain instantly.
*   Smooth the transition over 1-2 seconds to avoid "pumping" or flashing colors.

### Optimization 2: Face-Priority AWB

*   If a face is detected (AI/CV), weight the AWB statistics in the face region higher.
*   Ensures skin tones are correct even if the background is a weird color (e.g., green wall).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the "Planckian Locus"?** (The curve on the chromaticity diagram representing black body radiators/natural light sources).
2.  **Why is "Illuminant A" difficult for sensors?** (It has very little Blue energy. The Blue channel gain must be very high, causing noise).
3.  **What is "Color Casting"?** (A global tint overlaying the image).
4.  **Difference between AWB and CCM?** (AWB balances the White Point. CCM rotates the colors to match sRGB).

### Practical Challenges

1.  **Implement "Grey World" AWB:** Write a simple Python script that calculates `Gain_R = Mean_G / Mean_R` and `Gain_B = Mean_G / Mean_B` and applies it.
2.  **Test "Golden Hour":** Capture images at sunset. Ensure AWB preserves the warm mood and doesn't try to make the sunset look white/gray.

---

## 📚 Further Reading & Resources

### Standards
*   **CIE 1931 & 1976 Color Spaces.**

---

## 🎓 Summary

Today we covered:
- ✅ **Illuminants:** D65, A, TL84.
- ✅ **AWB:** Gray neutrality vs Preference.
- ✅ **Shading:** Color cast at corners.
- ✅ **Mixed Light:** The ultimate challenge.
- ✅ **Skin:** The most important color.

**Next:** Day 79 - Automotive Standards (AEC-Q100, ISO 16750).

---

**Day 78 Complete** | Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance
