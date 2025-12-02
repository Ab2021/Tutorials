# Day 171: EMC/EMI Testing
## Phase 3: Camera Systems & ISP | Week 27: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Understand** Electromagnetic Compatibility (EMC): Emission (EMI) vs Immunity (EMS).
2.  **Identify** sources of noise in camera systems: MIPI Clock, Switching Regulators, FPD-Link/GMSL cables.
3.  **Design** for EMC: Grounding, Shielding, Differential Pairs, Spread Spectrum Clocking (SSC).
4.  **Analyze** Test Reports: CISPR 25 (Automotive) or FCC Part 15 (Consumer).
5.  **Debug** EMI failures using a Near-Field Probe and Spectrum Analyzer.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Spectrum Analyzer (or SDR dongle for basics), Near-Field Probes.
*   **Context:** Why does the radio buzz when I turn on the dashcam?

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Emission vs Immunity
*   **Radiated Emission (RE):** Noise radiating through the air (Antenna effect).
*   **Conducted Emission (CE):** Noise traveling through power cables.
*   **Radiated Immunity (RI):** Can the camera survive a strong radar/radio signal nearby?
*   **Conducted Immunity (CI):** Can the camera survive noise on the power line?

### 🔹 Part 2: The Culprits
*   **MIPI CSI-2:** High speed clock (e.g., 1.5 GHz). Radiates if traces are not length-matched or impedance controlled.
*   **DC-DC Converters:** Switch at 1-2 MHz. Creates harmonics (2, 3, 4 MHz...).
*   **Coax Cables:** If the shield is broken or ground is poor, the cable becomes a giant antenna.

### 🔹 Part 3: CISPR 25 (Automotive Standard)
*   Strict limits to protect on-board radio receivers (FM, GPS, Bluetooth).
*   **Classes:** Class 1 (Relaxed) to Class 5 (Strict). OEMs usually demand Class 5.

---

## 💻 Implementation Examples

### Example 1: Spread Spectrum Clocking (SSC)

Modulating the clock frequency to spread the energy peak.

```c
// In Sensor Driver (e.g., IMX390)
// Register settings to enable SSC on MIPI PHY
static const struct imx390_reg ssc_settings[] = {
    {0x3000, 0x01}, // Enable SSC
    {0x3001, 0x05}, // Modulation Rate (e.g., 30kHz)
    {0x3002, 0x10}, // Modulation Depth (e.g., +/- 1%)
};

// Effect:
// Instead of a sharp spike at 1.5 GHz (Fail),
// you get a wider, lower hump from 1.485 GHz to 1.515 GHz (Pass).
```

### Example 2: PCB Layout Best Practices (Checklist)

1.  **Solid Ground Plane:** No cuts under high-speed traces.
2.  **Stitching Vias:** Connect Top Ground to Bottom Ground every 5mm along the edge of the board (Faraday Cage).
3.  **Differential Pairs:** Keep MIPI D+ and D- tightly coupled. Any separation creates a loop antenna.
4.  **Decoupling Caps:** Place 0.1uF caps as close as possible to power pins.

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Near-Field Probing

**Objective:** Find the noise source.

**Steps:**
1.  Power up the camera.
2.  Connect a Loop Probe to the Spectrum Analyzer.
3.  Hover over the PCB.
4.  **Observe:**
    *   **1.5 GHz:** MIPI Clock.
    *   **24 MHz:** Oscillator.
    *   **2 MHz:** Buck Converter.
5.  **Action:** If 1.5 GHz is leaking from the connector, add shielding tape.

### Lab 2: Cable Shielding Test

**Objective:** Good vs Bad cable.

**Steps:**
1.  Use an unshielded twisted pair. Observe high noise floor.
2.  Wrap it in Aluminum Foil and ground the foil.
3.  **Observe:** Noise drops by 20dB.
4.  **Lesson:** Coax/STP (Shielded Twisted Pair) is mandatory for SerDes.

### Lab 3: Ferrite Bead Magic

**Objective:** Filter Conducted Emission.

**Steps:**
1.  Measure noise on the 12V power line.
2.  Insert a Ferrite Bead in series.
3.  **Observe:** High frequency noise is absorbed (turned into heat).

---

## 🐛 Debugging EMC Failures

### Debug 1: "Failing FM Band (88-108 MHz)"

**Symptom:** CISPR 25 Class 5 Failure in FM range.

**Cause:**
*   Harmonics of the System Clock (e.g., 24 MHz x 4 = 96 MHz).
*   **Fix:** Change the crystal frequency slightly (e.g., to 25 MHz -> 100 MHz, moving it out of the station you are testing). Or use SSC.

### Debug 2: "Flickering Video when Walkie-Talkie used"

**Symptom:** Radiated Immunity failure.

**Cause:**
*   The reset line (RST) trace is long and acts as an antenna. The RF energy resets the sensor.
*   **Fix:** Add a 100pF capacitor on the RST line to ground (Low-pass filter).

---

## ⚡ Performance Optimization

### Optimization 1: Slew Rate Control

*   Slow down the rising/falling edges of digital signals (GPIOs, I2C).
*   Sharp edges contain infinite harmonics. Slower edges = Less EMI.
*   Configurable in many SoCs (`drive-strength` in Device Tree).

### Optimization 2: Common Mode Choke (CMC)

*   Place a CMC on the MIPI lines / Power lines.
*   Allows differential signals (Data) to pass. Blocks common mode noise (EMI).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Impedance Matching"?** (Ensuring Source = Load = 50 Ohms. Mismatch causes reflections, which radiate as EMI).
2.  **Why is a "Pigtail" ground bad?** (A long ground wire has inductance. It acts as an antenna. Ground connections should be short and wide).
3.  **What is "Quasi-Peak" detection?** (A measurement mode that weighs noise based on how annoying it is to the human ear. Pulsed noise is worse than constant noise).

### Practical Challenges

1.  **Design a Shield:** Design a sheet metal can to cover the PMIC and Inductor area. Solder it to the ground ring.
2.  **Review a Layout:** Look at a Gerber file. Find the "Return Path" for the high-speed current. Is it broken by a via or a slot?

---

## 📚 Further Reading & Resources

### Documentation
*   **"High Speed Digital Design: A Handbook of Black Magic" (Johnson & Graham).**
*   **Texas Instruments EMI Design Guidelines.**

---

## 🎓 Summary

Today we covered:
- ✅ **EMI/EMS:** Noise out / Noise in.
- ✅ **CISPR 25:** The automotive bar.
- ✅ **Sources:** Clocks and Switchers.
- ✅ **Fixes:** Shielding, Filtering, SSC.
- ✅ **Probing:** Finding the hotspot.

**Next:** Day 172 - Production Testing (EOL).

---

**Day 171 Complete** | Phase 3: Camera Systems & ISP | Week 27: Testing, Validation & Compliance


