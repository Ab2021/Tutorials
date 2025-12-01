# Day 33: Flash & Strobe Control
## Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control

---

## 🎯 Learning Objectives
1.  **Understand** the difference between Xenon (Strobe) and LED Flash.
2.  **Implement** the Pre-Flash Metering sequence for exposure calculation.
3.  **Synchronize** the sensor exposure window with the flash pulse (Rolling Shutter issues).
4.  **Develop** Red-Eye Reduction logic.
5.  **Control** Flash Color Temperature (Dual-Tone LED).
6.  **Debug** "Flash Banding" and synchronization latency.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with Flash Driver (GPIO/I2C) and LED.
*   **Software:** C/C++ Compiler.
*   **Knowledge:** Inverse Square Law ($Intensity \propto 1/Distance^2$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Flash Technologies
*   **Xenon:** High voltage discharge. Extremely bright, extremely short duration (10-100us). Freezes motion perfectly. Requires bulky capacitor.
*   **LED:** Constant current source. Lower brightness, longer duration (10ms - 100ms). Used in all smartphones. Can be used as "Torch" (Video Light).

### 🔹 Part 2: The Synchronization Problem
*   **Global Shutter (Xenon):** Easy. Open shutter, fire flash, close shutter.
*   **Rolling Shutter (LED):** The sensor exposes line-by-line.
    *   If Flash duration < Frame Readout Time, only a *band* of the image will be lit ("Flash Banding").
    *   **Requirement:** Flash duration must cover the *entire* readout time (e.g., 33ms for 30fps) OR we must use "Long Exposure" where all lines are open simultaneously (if possible).

### 🔹 Part 3: Pre-Flash Metering (TTL)
We don't know the subject distance or reflectivity.
1.  **Pre-Flash:** Fire a low-power pulse.
2.  **Measure:** Capture a frame. Compare brightness to Ambient frame.
3.  **Calculate:** Determine how much power is needed to reach Target Luma.
4.  **Main Flash:** Fire calculated power.

---

## 💻 Implementation Examples

### Example 1: Flash Driver (LED)

Controlling a Dual-Tone (Warm + Cool) LED driver via I2C (e.g., LM3644).

```cpp
/**
 * @brief Configure LED Flash
 * @param current_ma Total current in mA
 * @param temp_k Desired Color Temperature (e.g., 4500K)
 */
void set_flash_power(int current_ma, int temp_k) {
    // Dual Tone Logic
    // Cool LED (6000K), Warm LED (3000K)
    
    float ratio = (temp_k - 3000.0f) / (6000.0f - 3000.0f); // 0.0 = Warm, 1.0 = Cool
    ratio = std::clamp(ratio, 0.0f, 1.0f);
    
    int cool_ma = (int)(current_ma * ratio);
    int warm_ma = current_ma - cool_ma;
    
    // Write I2C Registers
    i2c_write(REG_LED1_CURRENT, cool_ma);
    i2c_write(REG_LED2_CURRENT, warm_ma);
    i2c_write(REG_ENABLE, 0x01); // Strobe Mode
}
```

### Example 2: Pre-Flash Metering Algorithm

Calculating the required gain for the main flash.

```cpp
/**
 * @brief Calculate Main Flash Power
 * @param ambient_luma Brightness before flash
 * @param preflash_luma Brightness during pre-flash
 * @param target_luma Desired brightness
 * @return Multiplier for Flash Power
 */
float calculate_flash_gain(float ambient_luma, float preflash_luma, float target_luma) {
    // 1. Isolate Flash Contribution
    float flash_diff = preflash_luma - ambient_luma;
    
    if (flash_diff < 1.0f) {
        // Flash had no effect (Subject too far)
        return 1.0f; // Max power
    }
    
    // 2. Calculate Gap to Target
    float needed_luma = target_luma - ambient_luma;
    
    // 3. Calculate Ratio
    // If Pre-Flash (at Power P) gave Diff D
    // We need Power X to get Diff N
    // X = P * (N / D)
    
    float ratio = needed_luma / flash_diff;
    
    return ratio;
}
```

### Example 3: Red-Eye Reduction Sequence

Red-eye is caused by light reflecting off the retina (blood vessels) when the pupil is wide open in dark.
**Solution:** Fire a series of short flashes *before* the main shot to constrict the pupil.

```cpp
void sequence_red_eye_reduction() {
    // 1. Pulse Sequence
    for (int i = 0; i < 5; i++) {
        fire_flash(LOW_POWER); // 50ms pulse
        msleep(100);           // 100ms gap
    }
    
    // 2. Wait for pupil constriction
    msleep(200);
    
    // 3. Main Capture
    fire_flash(HIGH_POWER);
    capture_frame();
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Flash Banding (Rolling Shutter)

**Objective:** Observe synchronization failure.

**Steps:**
1.  Set Exposure Time to 1ms (very short).
2.  Set Flash Duration to 1ms.
3.  Trigger capture.
4.  **Observation:** Only a horizontal strip of the image is lit. The rest is dark.
5.  **Fix:** Increase Flash Duration to cover the entire frame readout (e.g., 33ms). Or use "Torch Mode".

### Lab 2: Inverse Square Law Verification

**Objective:** Verify light falloff.

**Steps:**
1.  Place subject at 1 meter. Fire Flash. Measure Luma (L1).
2.  Place subject at 2 meters. Fire Flash. Measure Luma (L2).
3.  **Theory:** $L2 \approx L1 / 4$.
4.  **Result:** If L2 is much darker, your flash is too weak for long range.

### Lab 3: Color Mixing

**Objective:** Match Flash CCT to Ambient.

**Steps:**
1.  Scene: Tungsten light (3000K).
2.  Fire Cool Flash (6000K).
3.  **Observation:** Subject looks blue/ghostly against warm background.
4.  **Fix:** Adjust Dual-Tone Flash to 3000K. Subject blends naturally.

---

## 🐛 Debugging Techniques

### Debug 1: "Washed Out" Faces

**Symptom:** Subject is completely white (clipped).

**Cause:**
*   Pre-flash metering failed (subject moved closer after pre-flash).
*   Minimum flash power is still too strong for close-up.
*   **Fix:** Reduce Analog Gain (ISO) when Flash is active.

### Debug 2: Flash not firing

**Symptom:** Image is dark.

**Cause:**
*   Battery low (Flash inhibited by PMIC).
*   Thermal throttling (LED too hot).
*   **Fix:** Check PMIC status registers before firing.

---

## ⚡ Performance Optimization

### Optimization 1: Hardware Strobe Pin

*   Don't toggle GPIO in software (jitter).
*   Connect the Sensor's **STROBE** output pin to the Flash Driver's **STROBE** input pin.
*   Program the Sensor to pulse the pin exactly when exposure happens.
*   **Result:** Microsecond-perfect sync.

### Optimization 2: Charge Pump Management

*   Flash draws high current (1A+).
*   **Optimization:** Ramp up current slowly to avoid crashing the battery voltage (Brownout).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does Red-Eye happen?**
2.  **What is "Guide Number" (GN) for a flash?**
3.  **Why is Xenon flash better for freezing motion than LED?**
4.  **How does "Rear Curtain Sync" differ from "Front Curtain Sync"?**

### Practical Challenges

1.  **Implement a "Torch Mode" for Video:** Smoothly ramp brightness up/down to avoid jarring transitions.
2.  **Design a safety timeout:** Ensure the flash turns off after 500ms even if the software crashes (Hardware Watchdog).

---

## 📚 Further Reading & Resources

### Datasheets
*   **Texas Instruments LM3644:** Dual LED Flash Driver.
*   **ams OSRAM:** LED datasheets (Spectral power distribution).

---

## 🎓 Summary

Today we covered:
- ✅ **Flash Types:** Xenon vs LED.
- ✅ **Synchronization:** Rolling shutter challenges.
- ✅ **Metering:** Pre-flash TTL logic.
- ✅ **Red-Eye:** Pupil constriction.
- ✅ **Color:** Dual-tone mixing.

**Next:** Day 34 - Week 5 Review & 3A Tuning Project.

---

**Day 33 Complete** | Phase 3: Camera Systems & ISP | Week 5: 3A Algorithms & Control
