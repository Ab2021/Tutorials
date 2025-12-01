# Day 30: Auto-Focus (AF) Algorithms
## Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control

---

## 🎯 Learning Objectives
1.  **Understand** the principles of Focus (Circle of Confusion, Depth of Field).
2.  **Implement** Contrast Detection AF (CDAF) using Focus Value (FV) metrics.
3.  **Analyze** Phase Detection AF (PDAF) concepts and sensor requirements.
4.  **Design** AF Search Strategies (Hill Climbing, Coarse-to-Fine).
5.  **Control** the Voice Coil Motor (VCM) lens actuator.
6.  **Debug** AF hunting and breathing issues.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with VCM (Focus) Actuator (e.g., IMX219/IMX477).
*   **Software:** C/C++ Compiler.
*   **Knowledge:** Optics (Lens formula), Digital Signal Processing (High Pass Filter).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Physics of Focus
*   **Lens Formula:** $1/f = 1/u + 1/v$.
*   **In Focus:** A point source maps to a point on the sensor.
*   **Out of Focus:** A point source maps to a circle (Circle of Confusion).
*   **VCM:** A spring-loaded magnetic actuator that moves the lens barrel forward/backward by applying current (DAC code).

### 🔹 Part 2: Contrast Detection AF (CDAF)
The most common method for basic cameras.
*   **Principle:** A sharp image has high local contrast (sharp edges). A blurred image has low contrast.
*   **Metric:** We calculate a "Focus Value" (FV) which is essentially the energy of high frequencies in the image.
*   **Process:** Move lens -> Measure FV -> Repeat. Find the peak FV.
*   **Pros:** Accurate. Works on any sensor.
*   **Cons:** Slow. Requires "hunting" (moving past the peak to know it was the peak).

### 🔹 Part 3: Phase Detection AF (PDAF)
Used in DSLRs and modern smartphones.
*   **Principle:** Uses masked pixels (Left/Right) to measure the *direction* and *magnitude* of defocus.
*   **Pros:** Fast. Knows exactly where to move without hunting.
*   **Cons:** Requires special sensor hardware. Low light performance is worse than CDAF.

---

## 💻 Implementation Examples

### Example 1: Focus Value Calculation (CDAF)

We use a High Pass Filter (Laplacian or Sobel) to extract edges, then sum the squared values.

```cpp
/**
 * @brief Calculate Focus Value (FV)
 * @param img Luma Image (ROI only, e.g., center 25%)
 */
uint64_t calculate_focus_value(const Image8& img) {
    uint64_t fv = 0;
    int w = img.width;
    int h = img.height;
    
    // Simple 1D Gradient (Horizontal)
    // FV = Sum( (Pixel[x] - Pixel[x-1])^2 )
    // More robust: Laplacian
    
    #pragma omp parallel for reduction(+:fv)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            // Laplacian Kernel
            //  0 -1  0
            // -1  4 -1
            //  0 -1  0
            int val = 4 * img.data[y*w + x] 
                      - img.data[y*w + (x-1)] - img.data[y*w + (x+1)]
                      - img.data[(y-1)*w + x] - img.data[(y+1)*w + x];
            
            // Square to emphasize strong edges
            fv += (uint64_t)(val * val);
        }
    }
    
    // Normalize by number of pixels
    return fv / (w * h);
}
```

### Example 2: Hill Climbing Search Strategy

The standard CDAF algorithm.

```cpp
/**
 * @brief Hill Climbing AF
 * @return Best Lens Position (DAC Code)
 */
int af_hill_climbing(Camera& cam) {
    int current_pos = cam.get_lens_pos();
    int step = 10;
    int direction = 1; // 1 = Far to Near, -1 = Near to Far
    
    uint64_t prev_fv = 0;
    uint64_t curr_fv = 0;
    
    // 1. Measure Initial FV
    curr_fv = calculate_focus_value(cam.capture_preview());
    
    // 2. Start Moving
    for (int i = 0; i < 20; i++) { // Max steps safety
        prev_fv = curr_fv;
        
        // Move Lens
        current_pos += (step * direction);
        cam.set_lens_pos(current_pos);
        
        // Wait for settling (VCM mechanical delay)
        msleep(20); 
        
        // Measure New FV
        curr_fv = calculate_focus_value(cam.capture_preview());
        
        // 3. Check Direction
        if (curr_fv > prev_fv) {
            // We are going up the hill. Keep going.
        } else {
            // We went down the hill (past the peak).
            // Reverse direction and use smaller step
            direction *= -1;
            step /= 2;
            
            if (step < 2) {
                // We are close enough. Peak found.
                // Move back to previous best (approx)
                cam.set_lens_pos(current_pos + (step * direction));
                return current_pos;
            }
        }
    }
    return current_pos;
}
```

### Example 3: Coarse-to-Fine Search

Faster than simple hill climbing.
1.  **Coarse Scan:** Sweep the full range (0 to 1023) in large steps (e.g., 50).
2.  **Find Peak:** Identify the rough peak location.
3.  **Fine Scan:** Sweep around the rough peak in small steps (e.g., 5).

```cpp
int af_coarse_fine(Camera& cam) {
    // 1. Coarse Scan
    int best_coarse_pos = 0;
    uint64_t max_fv = 0;
    
    for (int pos = 0; pos <= 1000; pos += 50) {
        cam.set_lens_pos(pos);
        msleep(15);
        uint64_t fv = calculate_focus_value(cam.capture_preview());
        if (fv > max_fv) {
            max_fv = fv;
            best_coarse_pos = pos;
        }
    }
    
    // 2. Fine Scan
    // Scan +/- 50 around best_coarse_pos
    int start = std::max(0, best_coarse_pos - 50);
    int end = std::min(1000, best_coarse_pos + 50);
    
    int best_fine_pos = best_coarse_pos;
    
    for (int pos = start; pos <= end; pos += 5) {
        cam.set_lens_pos(pos);
        msleep(15);
        uint64_t fv = calculate_focus_value(cam.capture_preview());
        if (fv > max_fv) {
            max_fv = fv;
            best_fine_pos = pos;
        }
    }
    
    cam.set_lens_pos(best_fine_pos);
    return best_fine_pos;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: VCM Hysteresis Characterization

**Objective:** Understand mechanical backlash.

**Steps:**
1.  Place a target at a fixed distance.
2.  Scan lens from 0 -> 1023. Record Peak Position (e.g., 500).
3.  Scan lens from 1023 -> 0. Record Peak Position (e.g., 520).
4.  **Difference:** The difference (20 codes) is the Hysteresis.
5.  **Fix:** Always approach the final position from the same direction (e.g., always push Far->Near).

### Lab 2: Focus Breathing

**Objective:** Observe Field of View (FoV) change.

**Steps:**
1.  Set up a scene with objects at edges.
2.  Sweep focus from Near to Far.
3.  **Observation:** The image zooms in/out slightly. This is "Focus Breathing".
4.  **Impact:** Bad for video. Requires digital zoom compensation if severe.

### Lab 3: Low Light AF

**Objective:** Test AF failure.

**Steps:**
1.  Dim the lights.
2.  Run AF.
3.  **Observation:** Noise increases. The FV curve becomes noisy/flat. The peak is hard to find.
4.  **Fix:** Binning (combine pixels to reduce noise) or use an AF Assist Beam (LED).

---

## 🐛 Debugging Techniques

### Debug 1: Hunting (Pumping)

**Symptom:** Lens keeps moving back and forth without settling.

**Cause:**
*   FV curve is flat (low contrast scene).
*   Noise is higher than the FV peak difference.
*   **Fix:** Set a "Confidence Threshold". If Peak FV is not X% higher than Min FV, don't move.

### Debug 2: False Peak

**Symptom:** Focuses on the background instead of the subject.

**Cause:**
*   Background has higher contrast (e.g., blinds/fence) than the subject (smooth face).
*   **Fix:** Use Face Detection ROI. Only calculate FV inside the face box.

---

## ⚡ Performance Optimization

### Optimization 1: ROI-Based AF

*   Don't calculate FV for the whole 12MP image.
*   Use the center 25% (Center-Weighted) or a specific Touch AF ROI.
*   Reduces computation by 75%+.

### Optimization 2: Hardware Statistics (AF Stats)

*   ISPs have hardware engines that calculate FV (Sum of High Pass) for a grid (e.g., 16x12).
*   Read these stats registers instead of processing pixels in software.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does a high-pass filter measure sharpness?**
2.  **What is the advantage of PDAF over CDAF?**
3.  **Why do we need to wait (settle time) after moving the lens?**
4.  **How does Depth of Field affect AF accuracy?**

### Practical Challenges

1.  **Implement "Continuous AF" (CAF):** Monitor FV constantly. If it drops by > 10% (scene change), re-trigger AF.
2.  **Design a "Macro Mode" logic:** Limit the search range to the Near end (e.g., codes 500-1023).

---

## 📚 Further Reading & Resources

### Papers
*   **"Passive Autofocus Algorithms for Digital Cameras"** - Kehtarnavaz.

### Datasheets
*   **VCM Driver ICs:** DW9714, AD5820 (Study the I2C interface).

---

## 🎓 Summary

Today we covered:
- ✅ **Focus Physics:** Lens equation and VCM.
- ✅ **CDAF:** Using contrast (FV) to find sharpness.
- ✅ **Search Algorithms:** Hill Climbing vs Coarse-Fine.
- ✅ **PDAF:** The faster, hardware-based alternative.
- ✅ **Mechanical Issues:** Hysteresis and Breathing.

**Next:** Day 31 - 3A State Machine & Synchronization.

---

**Day 30 Complete** | Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control
