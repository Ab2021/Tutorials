# Day 29: Auto-White Balance (AWB) Algorithms
## Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control

---

## 🎯 Learning Objectives
1.  **Understand** the concept of Color Temperature (CCT) and Illuminants.
2.  **Implement** classic AWB algorithms: Gray World and White Patch.
3.  **Develop** advanced AWB: Mesh-based Statistics and Illuminant Estimation.
4.  **Visualize** the Planckian Locus on the CIE Chromaticity Diagram.
5.  **Tune** AWB for mixed lighting conditions.
6.  **Debug** color casts and AWB oscillation.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera, Grey Card, Macbeth Chart.
*   **Software:** C/C++ Compiler, Python.
*   **Knowledge:** Colorimetry (XYZ, xyY), Planck's Law.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Color Temperature
Objects emit light with a spectrum depending on their temperature (Black Body Radiation).
*   **Low CCT (2700K):** Warm, Red/Orange (Tungsten, Candle).
*   **High CCT (6500K):** Cool, Blue (Daylight, Overcast).
*   **Green/Magenta Tint:** Fluorescent lights don't follow the Black Body curve; they have spikes (Green). We need a 2D coordinate system (Temp + Tint).

### 🔹 Part 2: The Goal of AWB
The human brain adapts to lighting ("Color Constancy"). A white paper looks white under candle light and sunlight.
A camera sensor records the *reflected* spectrum = *Illuminant* x *Reflectance*.
*   **Goal:** Estimate the *Illuminant* and divide it out, so the image looks like it was taken under neutral (D65) light.
*   `R_balanced = R_raw * Gain_R`
*   `B_balanced = B_raw * Gain_B`
*   `G_balanced = G_raw` (Usually Green is the reference).

### 🔹 Part 3: Algorithms

#### 3.1 Gray World Assumption
*   **Theory:** The average color of a complex scene is neutral grey.
*   **Logic:** If Average(R) > Average(G), the light is reddish. Reduce R gain.
*   **Failure:** A scene with a giant red wall. AWB will make the wall grey (cyan cast).

#### 3.2 White Patch (Max RGB)
*   **Theory:** The brightest pixel in the image is a specular highlight (reflection of the light source), which is white.
*   **Logic:** Find Max(R), Max(G), Max(B). Scale gains so Max(R)=Max(G)=Max(B).
*   **Failure:** No white object in scene, or clipped highlights.

#### 3.3 Mesh-Based (Zone System)
*   **Theory:** Divide image into N x M zones. Analyze each zone. Discard zones that are likely "colored objects" (e.g., green grass, blue sky). Only use "Near Neutral" zones to estimate the illuminant.

---

## 💻 Implementation Examples

### Example 1: Gray World AWB

```cpp
/**
 * @brief Gray World AWB
 * @param stats Global Average R, G, B
 * @param r_gain Output Red Gain
 * @param b_gain Output Blue Gain
 */
void awb_gray_world(float avg_r, float avg_g, float avg_b, float& r_gain, float& b_gain) {
    // Avoid division by zero
    if (avg_r < 1e-5 || avg_b < 1e-5) {
        r_gain = 1.0f;
        b_gain = 1.0f;
        return;
    }
    
    // Normalize to Green
    r_gain = avg_g / avg_r;
    b_gain = avg_g / avg_b;
    
    // Clamp gains (Safety)
    // Typical range: 1.0 to 3.0 for R/B
    r_gain = std::clamp(r_gain, 1.0f, 4.0f);
    b_gain = std::clamp(b_gain, 1.0f, 4.0f);
}
```

### Example 2: Weighted Mesh AWB (Grey Edge Hypothesis)

We iterate over grid statistics. We calculate the "distance" of each block from the Planckian Locus (or a known Grey line). Blocks far away (saturated colors) get low weight.

```cpp
struct StatBlock {
    float r, g, b;
};

/**
 * @brief Mesh Based AWB
 */
void awb_mesh(const std::vector<StatBlock>& mesh, float& r_gain, float& b_gain) {
    float sum_r = 0, sum_g = 0, sum_b = 0;
    float total_weight = 0;
    
    for (const auto& block : mesh) {
        // 1. Calculate Chroma / Saturation
        // Simple approximation: |R-G| + |B-G|
        float sat = fabs(block.r - block.g) + fabs(block.b - block.g);
        
        // 2. Calculate Weight
        // Preference for low saturation (grey/white objects)
        // Weight = 1 / (1 + Saturation)
        float weight = 1.0f / (1.0f + sat * 0.1f);
        
        // 3. Filter Outliers (e.g., very dark or very bright)
        float luma = block.g;
        if (luma < 10 || luma > 250) weight = 0.0f;
        
        sum_r += block.r * weight;
        sum_g += block.g * weight;
        sum_b += block.b * weight;
        total_weight += weight;
    }
    
    if (total_weight > 0) {
        r_gain = sum_g / sum_r;
        b_gain = sum_g / sum_b;
    } else {
        // Fallback
        r_gain = 1.0f;
        b_gain = 1.0f;
    }
}
```

### Example 3: Temporal Smoothing

AWB shouldn't jump instantly if a red car drives by.

```cpp
/**
 * @brief Smooth AWB Gains
 */
void awb_smooth(float target_r, float target_b, float& current_r, float& current_b) {
    float speed = 0.05f; // Slow convergence
    
    current_r = current_r + speed * (target_r - current_r);
    current_b = current_b + speed * (target_b - current_b);
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Red Room" Failure

**Objective:** Demonstrate Gray World failure.

**Steps:**
1.  Point camera at a scene dominated by a red object (e.g., a red screen).
2.  Run Gray World.
3.  **Observation:** The red object becomes grey/desaturated. The background (if visible) turns Cyan.
4.  **Fix:** Use the Mesh AWB (Example 2). Since the red object has high saturation, its weight will be low. The AWB will ignore it and lock onto the small grey background.

### Lab 2: Illuminant Estimation Plot

**Objective:** Visualize where the light source falls.

**Steps:**
1.  Capture stats from a scene.
2.  Calculate `R/G` and `B/G` for every block.
3.  Plot these points on a 2D graph (X=R/G, Y=B/G).
4.  Overlay the "Planckian Locus" (the curve of valid white points).
5.  **Observation:** The "Grey" pixels will cluster *along* the curve. The colored pixels will be scattered away.
6.  **Algorithm:** Fit a line through the cluster to find the CCT.

### Lab 3: Mixed Lighting

**Objective:** Handle Indoor (Tungsten) + Outdoor (Daylight) mix.

**Steps:**
1.  Set up a scene with a window (Daylight) and a lamp (Tungsten).
2.  Observe the AWB. It will likely pick an average (e.g., 4000K).
3.  **Result:** Window looks Blue, Lamp looks Orange.
4.  **Advanced:** Dual-Illuminant AWB (Spatial). Apply different gains to different parts of the image (requires local AWB support in ISP).

---

## 🐛 Debugging Techniques

### Debug 1: Green/Magenta Cast

**Symptom:** Skin tones look sickly green or purple.

**Cause:**
*   Fluorescent lighting (Green spike).
*   AWB algorithm only tracking CCT (Red/Blue balance) and ignoring Tint (Green).
*   **Fix:** Adjust Green Gain relative to (R+B)/2. Or use a 2D (Temp/Tint) control loop.

### Debug 2: AWB Oscillation

**Symptom:** Color temperature keeps shifting Warm <-> Cool.

**Cause:**
*   Hysteresis is too small.
*   Noise in statistics.
*   **Fix:** Implement a "Stable Range". If CCT changes by < 200K, do not update gains.

---

## ⚡ Performance Optimization

### Optimization 1: Subsampling

*   We don't need every pixel to estimate the illuminant.
*   Use a 32x24 grid of averages (Hardware Stats).
*   Processing 768 points is instant compared to 12MP.

### Optimization 2: Integer Math for Gains

*   Gains are usually applied in the Raw domain (10-14 bit).
*   `Pixel_Out = (Pixel_In * Gain_Q8) >> 8`.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is Green usually kept at Gain = 1.0?**
2.  **What is the "Planckian Locus"?**
3.  **Why does Gray World fail on a "Green Field" scene?**
4.  **How does "Color Shading" (LSC) affect AWB?**

### Practical Challenges

1.  **Implement a "Daylight Lock" mode:** If the estimated CCT is > 5000K, force it to 5500K (Sunny).
2.  **Create a "Sunset Detection" logic:** If the scene is very bright AND very warm (2000K), do *not* correct it fully (preserve the mood).

---

## 📚 Further Reading & Resources

### Papers
*   **"Color Constancy: Research Website"** (Simon Fraser University).
*   **"Gray Edge Hypothesis"** - Van de Weijer.

### Tools
*   **Imatest:** Master module for AWB testing.

---

## 🎓 Summary

Today we covered:
- ✅ **Color Temperature:** Warm vs Cool.
- ✅ **Gray World:** The baseline assumption.
- ✅ **Mesh AWB:** Rejecting colored objects to find the true white.
- ✅ **Illuminant Estimation:** Finding the light source.
- ✅ **Tuning:** Handling mixed light and failure cases.

**Next:** Day 30 - Auto-Focus (AF) Algorithms.

---

**Day 29 Complete** | Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control
