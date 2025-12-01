# Day 18: SerDes Link Training, Equalization & Signal Integrity
## Phase 3: Camera Systems & ISP | Week 3: SerDes & Automotive Interfaces

---

## 🎯 Learning Objectives
1. **Understand** the physics of high-speed serial data transmission (Signal Integrity)
2. **Analyze** the Link Training process (Lock acquisition, EQ adaptation)
3. **Configure** Equalization settings (CTLE, DFE) for long cable runs
4. **Interpret** Eye Diagrams and Jitter measurements
5. **Implement** Spread Spectrum Clocking (SSC) for EMI compliance
6. **Diagnose** physical layer issues using internal tools (TDR, Eye Monitor)

---

## 📚 Prerequisites & Preparation
*   **Hardware:** SerDes EVKs, Coax Cables of various lengths (2m, 5m, 10m, 15m), Oscilloscope (optional but recommended)
*   **Software:** Register access tools (ALP, Maxim GUI)
*   **Knowledge:** Transmission Line Theory, Impedance Matching, Frequency Domain Analysis

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Physics of High-Speed Links

#### 1.1 The Channel
The "Channel" includes the PCB traces, connectors (FAKRA/HSD), and the cable itself.
*   **Insertion Loss (Attenuation):** Signal strength decreases with frequency and length. High-frequency components (edges) are attenuated more than low-frequency components.
*   **Return Loss (Reflection):** Impedance mismatches cause reflections, creating standing waves and closing the eye.
*   **Crosstalk (NEXT/FEXT):** Interference from adjacent lanes or cables.

#### 1.2 The "Eye" Diagram
The Eye Diagram is the superposition of all bits.
*   **Open Eye:** Good signal integrity. Distinct 1s and 0s.
*   **Closed Eye:** ISI (Inter-Symbol Interference) and Noise make it impossible to distinguish bits.
*   **Jitter:** Horizontal variation in edge timing.
*   **Noise:** Vertical variation in voltage levels.

### 🔹 Part 2: Equalization (EQ)

To recover a signal that looks like "mush" after 15 meters of cable, we use Equalization.

#### 2.1 Transmitter (TX) Pre-Emphasis
*   **Concept:** Boost the high-frequency components *before* transmission.
*   **Implementation:** Increase the amplitude of the first bit after a transition.
*   **Effect:** Counteracts the low-pass filter effect of the cable.

#### 2.2 Receiver (RX) Equalization
*   **CTLE (Continuous Time Linear Equalizer):** Analog filter that boosts high frequencies.
*   **DFE (Decision Feedback Equalizer):** Uses the history of previous bits to cancel out ISI from the current bit.
*   **Adaptive EQ (AEQ):** The receiver automatically adjusts CTLE/DFE coefficients during the Link Training phase to maximize the eye opening.

### 🔹 Part 3: Link Training Process

When a SerDes link powers up, it goes through a handshake:
1.  **Detection:** Deserializer detects a termination (50 Ohm) on the input.
2.  **Lock Acquisition:**
    *   Deserializer sweeps its AEQ settings.
    *   Checks for valid encoding (8b/10b or scrambling) and valid clock recovery.
    *   If valid, it asserts LOCK.
3.  **Tracking:** Once locked, the AEQ continuously makes small adjustments to track temperature and aging changes.

---

## 💻 Implementation Examples

### Example 1: Configuring TX Pre-Emphasis (GMSL)

If you have a very long cable and the AEQ is maxed out, you can help it by boosting the TX signal.

```c
/**
 * @file serdes_eq_config.c
 * @brief Configure TX Pre-emphasis and RX EQ
 */

#include <linux/module.h>
#include <linux/regmap.h>

/* MAX9295 Serializer Registers */
#define REG_TX_PREEMPHASIS  0x0042
#define REG_TX_AMPLITUDE    0x0043

/*
 * @brief Set TX Pre-emphasis
 * @param level: 0 (None) to 14 dB
 */
int gmsl_set_preemphasis(struct regmap *map, int level)
{
    unsigned int val;
    
    /* Map level to register value (simplified) */
    /* Usually a lookup table based on datasheet */
    if (level < 0) level = 0;
    if (level > 0xF) level = 0xF;
    
    /* Reg 0x0042: Pre-emphasis setting */
    /* Bit 7: Enable, Bit 3-0: Level */
    val = 0x80 | (level & 0x0F);
    
    return regmap_write(map, REG_TX_PREEMPHASIS, val);
}

/*
 * @brief Set TX Amplitude
 * Increase swing to improve SNR, but increases EMI and Power
 */
int gmsl_set_amplitude(struct regmap *map, int boost_mv)
{
    /* 0x00 = 100mV, ... 0x03 = 400mV (Example) */
    unsigned int val = 0x02; // Default
    
    if (boost_mv > 300) val = 0x03;
    
    return regmap_write(map, REG_TX_AMPLITUDE, val);
}
```

### Example 2: Monitoring RX Adaptive EQ (FPD-Link)

Monitoring the AEQ status is the best way to gauge link health.

```c
/**
 * @brief Monitor FPD-Link III AEQ Status
 */
struct link_health {
    int eq_level;     /* 0-14 */
    int cml_lock;     /* Boolean */
    int parity_errs;  /* Counter */
};

int ub954_get_health(struct regmap *map, int port, struct link_health *health)
{
    unsigned int val;
    
    /* Select Port */
    regmap_write(map, 0x4C, (1 << port));
    
    /* Read AEQ Status 1 (Reg 0xD2) */
    regmap_read(map, 0xD2, &val);
    
    /* Bits 2-0: EQ Level (First Stage) */
    health->eq_level = (val & 0x07);
    
    /* Read AEQ Status 2 (Reg 0xD3) */
    regmap_read(map, 0xD3, &val);
    /* Add Second Stage EQ if applicable */
    health->eq_level += ((val & 0x07) << 3);
    
    /* Check CML Lock (Reg 0x4D) */
    regmap_read(map, 0x4D, &val);
    health->cml_lock = (val & 0x01); // Lock Status
    
    return 0;
}
```

### Example 3: Spread Spectrum Clocking (SSC)

Enabling SSC spreads the energy of the clock harmonics, reducing peak EMI emissions.

```c
/**
 * @brief Enable Spread Spectrum on Serializer
 */
int gmsl_enable_ssc(struct regmap *map)
{
    /* 
     * Configure SSC Generator 
     * Spread: +/- 0.5%
     * Modulation Rate: 25 kHz
     */
    
    /* Reg 0x0400: SSC Config */
    /* Enable SSC, Center Spread */
    regmap_write(map, 0x0400, 0x10); // Enable
    
    /* Reg 0x0401: Deviation */
    regmap_write(map, 0x0401, 0x05); // +/- 0.5%
    
    return 0;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Long Cable" Challenge

**Objective:** Observe the effect of cable length on AEQ.

**Steps:**
1.  Connect a **2m** cable.
2.  Read the AEQ Status (EQ Level). Record it (e.g., Level 2).
3.  Connect a **15m** cable (or chain multiple cables).
4.  Read the AEQ Status. Record it (e.g., Level 12).
5.  **Analysis:** The receiver had to apply much more gain (boost) to recover the signal. If it hits the max level (e.g., 14 or 15), the link might be unstable.

### Lab 2: Eye Margin Analysis (Internal Eye Monitor)

Many modern SerDes chips (like UB954 or MAX9296) have a built-in "Eye Monitor" tool that can map the eye opening without an oscilloscope.

**Steps:**
1.  Use the vendor GUI (ALP or Maxim).
2.  Run "Eye Monitor" or "Margin Analysis".
3.  The tool will sweep the voltage and timing offsets and check for errors.
4.  **Result:** You get a 2D plot of the Eye Opening.
    *   **Pass:** The mask (central region) is clear of errors.
    *   **Fail:** The eye is collapsed or touches the mask.

### Lab 3: TDR (Time Domain Reflectometry)

**Objective:** Locate a fault in the cable.

**Steps:**
1.  Enable TDR mode on the SerDes chip (if supported).
2.  The chip sends a pulse and measures reflections.
3.  **Result:** It reports the distance to the fault (open or short).
    *   "Open detected at 12.5 meters".
    *   This saves hours of debugging in a vehicle!

---

## 🐛 Debugging Techniques

### Debug 1: "Sparkling" Video

**Symptom:** Random white/colored pixels (sparkles) in the image.

**Cause:** Bit errors on the link. The link is locked, but the BER (Bit Error Rate) is non-zero (e.g., 1e-9).

**Fix:**
1.  **Check EQ:** Is it maxed out?
2.  **Increase Amplitude:** Boost TX amplitude.
3.  **Check Connectors:** A loose FAKRA connector is a common culprit. Wiggle it and watch the error counter.

### Debug 2: EMI Failure

**Symptom:** The camera system fails EMC testing (radiates too much noise).

**Fix:**
1.  **Enable SSC:** Turn on Spread Spectrum.
2.  **Reduce Drive Strength:** Lower the GPIO/I2C drive strength on the Serializer if those traces are radiating.
3.  **Shielding:** Ensure the coax shield is properly grounded at both ends.

---

## ⚡ Performance Optimization

### Optimization 1: Manual EQ Tuning

Sometimes the Adaptive EQ gets "stuck" in a local minimum.
*   **Technique:** Force a specific EQ level that you know works well for your cable length, then allow it to adapt within a narrow range (+/- 2 levels).
*   **Benefit:** Faster lock time and more stability.

### Optimization 2: Power-Over-Coax (PoC) Filtering

*   **Issue:** The high-speed data and DC power share the same cable.
*   **Solution:** You need a high-quality PoC filter (Inductor/Ferrite) to separate them.
*   **Optimization:** Ensure the inductor's Self-Resonant Frequency (SRF) is higher than the Nyquist frequency of the data link (e.g., > 3 GHz for a 6 Gbps link).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do high frequencies attenuate more than low frequencies in a cable?**
2.  **What is the difference between Pre-emphasis (TX) and Equalization (RX)?**
3.  **How does Spread Spectrum Clocking reduce EMI?**
4.  **What does a "Closed Eye" indicate?**

### Practical Challenges

1.  **Interpret an Eye Diagram:** Given an eye height of 100mV and width of 0.6 UI, is this a good link? (Assume min req is 50mV / 0.3 UI).
2.  **Debug Scenario:** A 10m link works fine at room temperature but unlocks at -40°C. What parameter should you adjust?

---

## 📚 Further Reading & Resources

### Books
*   **"High-Speed Digital Design: A Handbook of Black Magic"** by Howard Johnson (The bible of Signal Integrity).
*   **"Signal Integrity and Power Integrity Simplified"** by Eric Bogatin.

### Tools
*   **Keysight ADS:** For simulating channel models.
*   **Vendor Tools:** TI ALP, Maxim SerDes GUI.

---

## 🎓 Summary

Today we covered:
- ✅ **Signal Integrity:** Insertion loss, return loss, and jitter.
- ✅ **Equalization:** TX Pre-emphasis and RX Adaptive EQ.
- ✅ **Link Training:** The handshake process to establish lock.
- ✅ **Diagnostics:** Eye diagrams, TDR, and error counters.
- ✅ **EMI:** Spread Spectrum Clocking.

**Next:** Day 19 - Virtual Channels & Aggregation (Deserializer configuration).

---

**Day 18 Complete** | Phase 3: Camera Systems & ISP | Week 3: SerDes & Automotive Interfaces
