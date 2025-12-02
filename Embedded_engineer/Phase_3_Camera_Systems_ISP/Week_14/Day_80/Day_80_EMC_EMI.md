# Day 80: EMC/EMI Testing for Camera Modules
## Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Understand** Electromagnetic Compatibility (EMC): Emissions (Don't be noisy) vs Immunity (Don't be sensitive).
2.  **Analyze** CISPR 25 Standard: Limits for Radiated and Conducted Emissions in vehicles.
3.  **Identify** Noise Sources: MIPI CSI-2 clock, DC-DC Converters, SerDes Link.
4.  **Implement** Mitigation Techniques: Spread Spectrum Clocking (SSC), Common Mode Chokes (CMC), Shielding.
5.  **Perform** Pre-compliance testing using a Spectrum Analyzer and Near-Field Probe.
6.  **Debug** "Flicker" caused by RF interference.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Spectrum Analyzer (or SDR), Near-Field Probes, Camera Module.
*   **Standards:** CISPR 25 (Emissions), ISO 11452 (Immunity).
*   **Knowledge:** Frequency Domain (FFT), Decibels (dBµV), Differential Signaling.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem
*   **Emissions:** A camera running a 1.5Gbps MIPI clock acts like a radio transmitter. If it radiates at 1575.42 MHz, it kills the car's GPS.
*   **Immunity:** If the car drives past a powerful TV tower, the RF energy might induce currents in the camera cable, causing the image to freeze.

### 🔹 Part 2: CISPR 25 (Emissions)
*   **Classes:** Class 1 (Loose) to Class 5 (Strict). OEMs usually demand Class 5.
*   **Bands:** LW, MW, FM, GPS, Bluetooth/WiFi.
*   **Conducted:** Noise traveling back up the power wire.
*   **Radiated:** Noise traveling through the air (Antenna).

### 🔹 Part 3: Noise Sources in Cameras
1.  **MIPI CSI-2:** High-speed differential clock. If the pair is not perfectly length-matched, it becomes a Common Mode antenna.
2.  **DC-DC Buck Converters:** Switching frequency (e.g., 2MHz) and its harmonics (4, 6, 8...).
3.  **Sensor Clock (MCLK):** 24MHz or 27MHz square wave. Rich in odd harmonics.

---

## 💻 Implementation Examples

### Example 1: Spread Spectrum Clocking (SSC)

Spreading the energy to lower the peak.

```c
/* In Sensor Driver (IMX219) */
static int imx219_enable_ssc(struct imx219 *sensor)
{
    /* 
     * SSC Modulation: +/- 0.5%
     * Modulation Freq: 30kHz
     */
    imx219_write_reg(sensor, 0x0300, 0x01); // Enable SSC
    imx219_write_reg(sensor, 0x0301, 0x05); // Spread Width
    
    return 0;
}
```
*   **Result:** A sharp peak at 750MHz becomes a wider, shorter "hump". Passes CISPR 25.

### Example 2: Common Mode Choke (CMC) Selection

For MIPI CSI-2 lines.

*   **Goal:** Pass Differential signals (Video), Block Common Mode signals (Noise).
*   **Spec:**
    *   Differential Impedance: 100 Ohm.
    *   Common Mode Attenuation: > 15dB at 700MHz.
    *   DC Resistance: Low (< 2 Ohm).
*   **Part:** Murata DLW series or TDK ACM series.

### Example 3: PCB Layout Best Practices

1.  **Solid Ground Plane:** Never route high-speed traces over a split plane.
2.  **Stitching Vias:** Connect GND planes every 5mm along the edge of the board (Faraday Cage).
3.  **Length Matching:** MIPI P/N pairs must be matched within 0.1mm to prevent Mode Conversion (Diff -> Common).

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Near-Field Probing

**Objective:** Find the noisy spot.

**Steps:**
1.  Power up the camera.
2.  Connect a Loop Probe to the Spectrum Analyzer.
3.  Hover over the PCB.
4.  **Observation:**
    *   **Buck Converter:** Strong low-frequency spikes (2MHz, 4MHz...).
    *   **MIPI Trace:** Strong high-frequency spikes (750MHz, 1.5GHz...).
5.  **Fix:** Add a shield can over the noisy area.

### Lab 2: Cable Shielding Test

**Objective:** Coax vs Unshielded.

**Steps:**
1.  Use a long (1m) unshielded ribbon cable for MIPI (if possible).
    *   **Result:** Massive radiation. GPS signal lost.
2.  Wrap the cable in Copper Tape (grounded at both ends).
    *   **Result:** Noise drops by > 20dB.
3.  **Lesson:** This is why GMSL uses Coax or Shielded Twisted Pair (STP).

### Lab 3: Immunity Test (Poor Man's BCI)

**Objective:** Can it survive a Walkie-Talkie?

**Steps:**
1.  Stream video.
2.  Key a handheld radio (5W) near the camera cable.
3.  **Observation:**
    *   **Good:** No effect.
    *   **Bad:** Video flickers, lines appear, or link drops.
4.  **Fix:** Add Ferrite Beads on the power input.

---

## 🐛 Debugging EMC Issues

### Debug 1: GPS Desense

**Symptom:** Car GPS loses lock when Camera is on.

**Cause:**
*   Camera radiating at 1575 MHz (Harmonic of 24MHz? No. Harmonic of 750MHz? Yes, 2nd harmonic = 1500. Close).
*   **Fix:** Shift the MIPI clock frequency slightly (e.g., change blanking) to move the harmonic away from 1575 MHz.

### Debug 2: "Comb" Spectrum

**Symptom:** Evenly spaced spikes every 24MHz.

**Cause:**
*   MCLK (Sensor Clock) ringing.
*   **Fix:** Add a Series Resistor (33 Ohm) on the MCLK line to dampen reflections (Source Termination).

---

## ⚡ Performance Optimization

### Optimization 1: Slew Rate Control

*   Slow down the rising/falling edges of digital signals (I2C, GPIO).
*   Sharper edges = More high-frequency harmonics.
*   Many SoCs/Sensors have "Drive Strength" registers. Set to minimum required.

### Optimization 2: Frequency Planning

*   Choose switching frequencies (DC-DC) that don't have harmonics in sensitive bands (AM Radio).
*   e.g., Don't switch at 530kHz (AM band start). Switch at 2.1MHz (Above AM).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is "Differential Signaling" good for EMC?** (The fields from P and N cancel each other out if balanced).
2.  **What is a "Faraday Cage"?** (A conductive enclosure that blocks EM fields).
3.  **Difference between "Class A" and "Class B" in FCC/CE?** (Class A = Industrial, Class B = Residential/Stricter).
4.  **Why do we ground the shield at *both* ends for HF noise?** (To create a return path for the noise current).

### Practical Challenges

1.  **Design a Shield Can:** Draw a footprint for a metal shield clip on the PCB around the PMIC and Inductor.
2.  **Calculate Harmonics:** If MCLK is 27MHz, list the first 5 harmonics. Do any fall into the FM Radio band (88-108MHz)? (3rd = 81, 4th = 108. Yes, 4th harmonic is a risk).

---

## 📚 Further Reading & Resources

### Standards
*   **CISPR 25 Ed. 4.**
*   **ISO 11452-4 (BCI).**

---

## 🎓 Summary

Today we covered:
- ✅ **EMC:** Emissions vs Immunity.
- ✅ **CISPR 25:** The limit lines.
- ✅ **Sources:** Clocks, Switchers, SerDes.
- ✅ **Fixes:** SSC, Chokes, Shielding.
- ✅ **Probing:** Finding the hot spots.

**Next:** Day 81 - Production Line Testing (EOL).

---

**Day 80 Complete** | Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance
