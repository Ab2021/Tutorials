# Day 149: Lighting Conditions (Night, Glare)
## Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases

---

> **📝 Day 149 Focus:**
> Cameras are like human eyes: they struggle when it's too dark (Night) or too bright (Sun Glare). **Lighting** variation is the most common cause of perception failure. Today, we learn to normalize images to handle these extremes.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** Dynamic Range and the problem of under/over-exposure.
2.  **Implement** Gamma Correction and Histogram Equalization (CLAHE).
3.  **Perform** HDR (High Dynamic Range) merging of multiple exposures.
4.  **Detect** and mitigate Sun Glare regions.
5.  **Train** models with brightness/contrast augmentation.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Image Processing:** Histograms, Pixel intensity (0-255).
-   **Camera Physics:** Exposure time, ISO, Aperture.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `opencv-python`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Dynamic Range Problem

-   **Night:** Signal-to-Noise Ratio (SNR) drops. Motion blur increases (longer exposure).
-   **Glare:** Direct sunlight or headlights saturate pixels (255, 255, 255). Information is lost (Clipping).
-   **Tunnel Exit:** Classic HDR case. Dark inside, bright outside. Standard cameras can't see both.

### 🔹 Part 2: Enhancement Techniques

1.  **Gamma Correction:** Non-linear mapping ($V_{out} = V_{in}^\gamma$).
    -   $\gamma < 1$: Brightens dark areas (Night).
    -   $\gamma > 1$: Darkens bright areas.
2.  **Histogram Equalization:** Spreads out intensity distribution to use full range.
    -   **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Local equalization. Prevents noise amplification in flat regions.
3.  **HDR:** Combine Short Exposure (highlights) + Long Exposure (shadows).

---

## 💻 Implementation: Low-Light Enhancement

**Scenario:**
-   Input: A dark image of a road at night.
-   Task: Make obstacles visible.

### 🛠️ Setup
Create `week22_day149` and `light_enhance.py`.

```bash
mkdir -p ~/ros2_ws/src/week22_day149
cd ~/ros2_ws/src/week22_day149
touch light_enhance.py
```

### 👨‍💻 Code: CLAHE & Gamma

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(img, gamma=1.0):
    # Lookup Table
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_clahe(img):
    # Convert to LAB color space (L = Lightness)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def detect_glare(img):
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold (Saturated pixels)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Dilate to cover bloom
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask

def main():
    # 1. Create Synthetic Dark Image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (20, 20, 20) # Very dark
    cv2.rectangle(img, (200, 200), (400, 300), (50, 50, 50), -1) # Dark object
    
    # 2. Create Synthetic Glare Image
    glare_img = np.zeros((400, 600, 3), dtype=np.uint8)
    glare_img[:] = (100, 100, 100)
    cv2.circle(glare_img, (300, 100), 50, (255, 255, 255), -1) # Sun
    cv2.circle(glare_img, (300, 100), 80, (255, 255, 255), -1) # Bloom (simulated)
    
    # 3. Process
    gamma_img = gamma_correction(img, gamma=0.4) # Brighten
    clahe_img = apply_clahe(img)
    glare_mask = detect_glare(glare_img)
    
    # 4. Visualize
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Original (Dark)")
    
    plt.subplot(2, 3, 2)
    plt.imshow(gamma_img)
    plt.title("Gamma (0.4)")
    
    plt.subplot(2, 3, 3)
    plt.imshow(clahe_img)
    plt.title("CLAHE")
    
    plt.subplot(2, 3, 4)
    plt.imshow(glare_img)
    plt.title("Original (Glare)")
    
    plt.subplot(2, 3, 5)
    plt.imshow(glare_mask, cmap='gray')
    plt.title("Glare Mask")
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: HDR Merging

### Lab Objectives
1.  **Capture:** Take 3 photos of a window (indoors looking out).
    -   Underexposed (Dark room, Window clear).
    -   Normal.
    -   Overexposed (Room clear, Window white).
2.  **Merge:**
    -   Use `cv2.createMergeDebevec()` or `cv2.createMergeMertens()`.
    -   **Observation:** The result shows details in *both* the room and the window.
    -   **Relevance:** Autonomous cars use HDR cameras (120dB+) to see into tunnels and shadows simultaneously.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Noise Amplification
**Symptom:** Enhanced night image looks grainy/speckled.
**Cause:** Gamma/CLAHE boosts noise along with signal.
**Solution:** Apply Denoising (Gaussian Blur or Bilateral Filter) *before* enhancement.

#### 2. Color Shift
**Symptom:** Colors look unnatural after enhancement.
**Cause:** Applying equalization to RGB channels independently changes the ratio (Hue).
**Solution:** Convert to LAB or HSV. Apply enhancement only to L/V channel. Preserve A/B or H/S.

---

## ⚡ Optimization & Best Practices

### 1. Auto-Exposure Control (AEC)
Don't rely just on post-processing.
-   Control the camera hardware.
-   **ROI-based AEC:** Set exposure based on the *road* region, ignoring the sky.

### 2. Sun Glare Handling
If a region is saturated (Glare Mask):
-   **Tracking:** Don't try to detect objects there.
-   **Planning:** Treat it as an "Unknown/Blind" zone. Slow down.
-   **Hardware:** Use Polarizing filters.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between Global and Adaptive Histogram Equalization?
    *   **A:** Global uses the whole image (can wash out details). Adaptive (CLAHE) uses small tiles (preserves local contrast).
2.  **Q:** Why is HDR important for ADAS?
    *   **A:** Real-world scenes have dynamic ranges ($10^5:1$) that exceed standard sensors ($10^3:1$). HDR captures the full range.
3.  **Q:** How does Gamma Correction work?
    *   **A:** It expands the dark values and compresses the bright values (if $\gamma < 1$), making shadows visible.

### Challenge Task
**Task:** Night Lane Detection.
1.  Take a night video.
2.  Apply CLAHE.
3.  Run Canny Edge Detection.
4.  Compare with Canny on original image.
5.  Observe significantly better edge continuity.

---

## 📚 Further Reading & References
-   [OpenCV HDR Tutorial](https://docs.opencv.org/4.x/d2/df0/tutorial_py_hdr.html)
-   [Sony IMX Sensors (HDR)](https://www.sony-semicon.co.jp/e/products/IS/automotive/)

---

**Day 149 Complete** | Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases
