# Day 28: Auto-Exposure (AE) Control Loop Design
## Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control

---

## 🎯 Learning Objectives
1.  **Understand** the AE Control Loop dynamics (Feedback System).
2.  **Implement** a PID-based AE Controller to maintain target brightness.
3.  **Design** an Exposure Table (Program Line) balancing Gain vs Integration Time.
4.  **Handle** boundary conditions (Min/Max Gain, Flicker steps).
5.  **Simulate** the AE loop response to sudden lighting changes.
6.  **Debug** AE oscillation and hunting issues.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with manual exposure/gain control.
*   **Software:** C/C++ Compiler, Python (for simulation).
*   **Knowledge:** Control Theory (PID), Radiometry (EV, Lux).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The AE Feedback Loop
Auto-Exposure is a classic Closed-Loop Control System.
*   **Set Point (SP):** Target Luma (e.g., 50% grey or code value 128).
*   **Process Variable (PV):** Current Average Luma (measured from statistics).
*   **Error:** $E = SP - PV$.
*   **Control Output:** New Exposure ($Gain \times Time$).

### 🔹 Part 2: Exposure Value (EV) & Program Line
We can achieve the same brightness with different combinations of Gain and Time.
*   $Exposure = Gain \times IntegrationTime$.
*   **Program Line:** A strategy to traverse this space.
    *   *Strategy A (Low Noise):* Maximize Time first, then increase Gain.
    *   *Strategy B (Action):* Limit Time (to freeze motion), then increase Gain.
    *   *Strategy C (Flicker Free):* Lock Time to grid (1/100s, 1/120s), use Gain for fine tuning.

### 🔹 Part 3: Loop Dynamics
*   **Damping:** We don't want to jump to the target instantly (looks jerky). We want a smooth convergence.
*   **Hysteresis:** Don't change exposure if the error is small (prevents "breathing").
*   **Stability:** Avoid oscillation (hunting) around the target.

---

## 💻 Implementation Examples

### Example 1: AE Statistics Collection

We need to measure the current brightness. A simple average is often not enough (fooled by bright windows). We use a Weighted Average.

```cpp
/**
 * @brief Calculate Weighted Luma
 * @param stats 16x12 Grid of Luma averages (from ISP hardware)
 * @param weights 16x12 Grid of weights (Center weighted)
 */
float calculate_current_luma(const float* stats, const float* weights, int w, int h) {
    float sum_luma = 0;
    float sum_weight = 0;
    
    for (int i = 0; i < w * h; i++) {
        sum_luma += stats[i] * weights[i];
        sum_weight += weights[i];
    }
    
    return sum_luma / sum_weight;
}
```

### Example 2: The AE Controller (Damped)

A simple Proportional controller with damping.

```cpp
/**
 * @brief AE Control Step
 * @param current_luma Measured brightness (0-255)
 * @param target_luma Desired brightness (e.g., 128)
 * @param current_exposure Current Total Exposure (Gain * Time)
 * @return New Total Exposure
 */
float run_ae_control(float current_luma, float target_luma, float current_exposure) {
    // 1. Calculate Ratio (Error)
    // Avoid division by zero
    if (current_luma < 1.0f) current_luma = 1.0f;
    
    float ratio = target_luma / current_luma;
    
    // 2. Apply Damping (Low Pass Filter)
    // We don't apply the full ratio instantly.
    // New_Ratio = 1.0 + Speed * (Target_Ratio - 1.0)
    float speed = 0.2f; // Convergence speed (0.0 to 1.0)
    float damped_ratio = 1.0f + speed * (ratio - 1.0f);
    
    // 3. Calculate New Exposure
    float new_exposure = current_exposure * damped_ratio;
    
    // 4. Hysteresis (Deadband)
    // If change is less than 5%, ignore it to prevent breathing
    if (fabs(damped_ratio - 1.0f) < 0.05f) {
        return current_exposure;
    }
    
    return new_exposure;
}
```

### Example 3: Exposure Decomposition (Program Line)

Convert "Total Exposure" into "Gain" and "Time".

```cpp
struct ExposureSetting {
    float gain; // Analog Gain (e.g., 1.0x to 16.0x)
    int time_us; // Integration Time in microseconds
};

/**
 * @brief Decompose Exposure according to "Low Noise" strategy
 * @param total_exposure Target Gain * Time
 */
ExposureSetting decompose_exposure(float total_exposure) {
    ExposureSetting out;
    
    // Constraints
    const int MIN_TIME = 100;    // 100us
    const int MAX_TIME = 33333;  // 33ms (30fps)
    const float MIN_GAIN = 1.0f;
    const float MAX_GAIN = 16.0f;
    
    // Strategy: Maximize Time first (to keep Gain low)
    
    // 1. Try with Min Gain
    float needed_time = total_exposure / MIN_GAIN;
    
    if (needed_time <= MAX_TIME) {
        // We can achieve it with Min Gain
        out.time_us = (int)std::max((float)MIN_TIME, needed_time);
        out.gain = MIN_GAIN;
    } else {
        // We need more than Max Time, so we must increase Gain
        out.time_us = MAX_TIME;
        float needed_gain = total_exposure / MAX_TIME;
        out.gain = std::min(MAX_GAIN, needed_gain);
    }
    
    return out;
}
```

### Example 4: Flicker Avoidance

If we are in a 50Hz country, lights flicker at 100Hz (10ms period).
Exposure time MUST be a multiple of 10ms (10ms, 20ms, 30ms) to integrate full cycles and avoid banding.

```cpp
/**
 * @brief Quantize Time for Flicker Avoidance
 */
int quantize_time_flicker(int time_us, int flicker_freq) {
    int period_us = 1000000 / (flicker_freq * 2); // 10000us for 50Hz
    
    // Find nearest multiple
    int steps = (time_us + period_us / 2) / period_us;
    if (steps < 1) steps = 1;
    
    return steps * period_us;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: AE Convergence Simulation

**Objective:** Visualize how the AE loop settles.

**Steps:**
1.  Write a Python script simulating a scene.
    *   `Scene_Brightness = 1000` (Lux).
    *   `Camera_Luma = Scene_Brightness * Exposure`.
2.  Implement the AE Controller loop.
3.  **Step Change:** At Frame 50, drop `Scene_Brightness` to 100.
4.  **Plot:** `Luma` vs `Frame`.
5.  **Tune:** Adjust `speed` parameter.
    *   High speed: Fast settling, potential overshoot.
    *   Low speed: Smooth, slow settling.

### Lab 2: Metering Modes

**Objective:** Implement Center-Weighted vs Spot Metering.

**Steps:**
1.  Create a synthetic image with a bright window in the center and dark corners.
2.  **Spot Metering:** Use only the center 10% stats. Result: Window is exposed correctly, corners are black.
3.  **Matrix Metering:** Average the whole image. Result: Window is blown out, corners are visible.
4.  **Highlight Weighted:** Look at the Histogram. Ensure top 5% of pixels are not clipped.

### Lab 3: Flicker Step

**Objective:** Observe banding.

**Steps:**
1.  Point camera at a fluorescent light (or PWM LED).
2.  Set Exposure Time to 8ms (not a multiple of 10ms/8.33ms).
3.  **Observation:** Rolling bands (dark/light stripes) moving across the image.
4.  Set Exposure Time to 10ms (for 50Hz) or 8.33ms (for 60Hz).
5.  **Observation:** Bands disappear.

---

## 🐛 Debugging Techniques

### Debug 1: AE Oscillation

**Symptom:** Brightness goes Up-Down-Up-Down constantly.

**Cause:**
*   **Latency:** The stats we read *now* are from a frame captured *2 frames ago*. If we react too fast, we overshoot.
*   **Fix:** Reduce convergence speed. Account for pipeline delay (Pipeline Depth).

### Debug 2: Exposure "Breathing"

**Symptom:** Brightness changes slightly even when scene is static.

**Cause:**
*   Noise in the stats causes small error fluctuations.
*   **Fix:** Increase Hysteresis (Deadband). Don't change exposure if error < 5%.

---

## ⚡ Performance Optimization

### Optimization 1: Predictive AE

*   If we know the scene is changing linearly (e.g., panning from dark to bright), we can predict the next Luma and adjust exposure *proactively* rather than reactively.
*   Kalman Filter can be used to estimate the scene brightness trend.

### Optimization 2: Hardware Statistics

*   Don't calculate average Luma in software (CPU).
*   Use the ISP's hardware statistics engine (Histogram / Grid Stats) which is free.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is the AE loop considered a "Delayed Feedback" system?**
2.  **What is the "Sunny 16" rule?**
3.  **Why does increasing Gain increase Noise?**
4.  **How does "Backlight Compensation" work?**

### Practical Challenges

1.  **Implement a "Manual Mode" override** in your AE controller that accepts user inputs but respects min/max hardware limits.
2.  **Design a "Night Mode" program line** that allows FPS to drop (increase Max Time) to keep Gain low.

---

## 📚 Further Reading & Resources

### Standards
*   **APEX System:** Additive System of Photographic Exposure.

### Papers
*   **"Camera Auto-Exposure Control for Surveillance Systems"** - IEEE.

---

## 🎓 Summary

Today we covered:
- ✅ **AE Loop:** Setpoint, Error, Output.
- ✅ **Exposure Triangle:** Gain, Time, Aperture (fixed in mobile).
- ✅ **Program Line:** Strategy for traversing exposure space.
- ✅ **Flicker:** The importance of grid-aligned timing.
- ✅ **Stability:** Damping and Hysteresis.

**Next:** Day 29 - Auto-White Balance (AWB) Algorithms.

---

**Day 28 Complete** | Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control
