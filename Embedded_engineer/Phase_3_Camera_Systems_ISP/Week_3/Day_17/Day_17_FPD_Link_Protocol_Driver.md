# Day 17: FPD-Link III/IV Protocol & Driver Implementation
## Phase 3: Camera Systems & ISP | Week 3: SerDes & Automotive Interfaces

---

## 🎯 Learning Objectives
1. **Understand** the FPD-Link III and IV architecture (Texas Instruments)
2. **Compare** FPD-Link vs GMSL protocols and feature sets
3. **Implement** FPD-Link serializer/deserializer initialization (DS90UB953/954)
4. **Configure** bidirectional control channels and GPIO forwarding
5. **Develop** a driver for FPD-Link based multi-camera systems
6. **Debug** FPD-Link specific issues (Back Channel frequency, Adaptive EQ)

---

## 📚 Prerequisites & Preparation
*   **Hardware:** TI FPD-Link EVK (DS90UB953 Serializer, DS90UB954 Deserializer)
*   **Software:** Linux Kernel Driver Development environment
*   **Datasheets:** TI DS90UB953-Q1, DS90UB954-Q1 (Essential)
*   **Tools:** Analog LaunchPAD (ALP) software (optional but helpful)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: FPD-Link Architecture

#### 1.1 FPD-Link III Overview
FPD-Link III (Flat Panel Display Link) is TI's proprietary SerDes interface, widely used in automotive infotainment and ADAS.

*   **Forward Channel:** High-speed video data (up to 4 Gbps per lane).
*   **Back Channel:** Low-speed control data (embedded in the same cable).
*   **Bidirectional Control:** Supports I2C, GPIO, SPI, and Interrupts.
*   **Cabling:** Coax (Single-ended) or STP (Differential).

#### 1.2 FPD-Link III vs GMSL
| Feature | FPD-Link III (TI) | GMSL2 (Maxim/ADI) |
| :--- | :--- | :--- |
| **Encoding** | Proprietary (Scrambled) | Scrambled |
| **Back Channel** | Lower speed (e.g., 50 Mbps) | High speed (187 Mbps) |
| **Sync** | Frame Sync via GPIO/BC | Frame Sync via GPIO/BC |
| **Daisy Chain** | Strong support (Ring/Chain) | Supported (Splitter) |
| **Ecosystem** | TI Processors (TDA4, Jacinto) | NVIDIA Jetson (Orin/Xavier) |

#### 1.3 FPD-Link IV (Next Gen)
*   **Bandwidth:** Up to 13.5 Gbps per lane.
*   **Architecture:** DSP-based receiver for robust signal integrity.
*   **Features:** Ethernet tunneling, dual CSI-2 ports, advanced diagnostics.

### 🔹 Part 2: DS90UB954 Deserializer Architecture

The DS90UB954 is a dual-port deserializer (Rx0, Rx1) that outputs to a single CSI-2 port.

*   **Rx Ports:** 2x FPD-Link III inputs (from 953 or 933 serializers).
*   **CSI-2 Output:** 4-lane MIPI CSI-2.
*   **Aggregation:** Can combine 2x 2MP cameras into one 4MP stream (Line Interleaved or VC based).
*   **Virtual Channels:** Supports re-mapping of incoming VCs.

---

## 💻 Implementation Examples

### Example 1: FPD-Link III Deserializer Driver (DS90UB954)

This driver initializes the UB954 and configures the RX ports.

```c
/**
 * @file ub954_driver.c
 * @brief Initialization for TI DS90UB954 Deserializer
 */

#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/regmap.h>
#include <linux/delay.h>

/* UB954 Registers */
#define UB954_REG_I2C_DEV_ID    0x00
#define UB954_REG_RESET         0x01
#define UB954_REG_RX_PORT_CTL   0x0C
#define UB954_REG_IO_CTL        0x0D
#define UB954_REG_BCC_CONFIG    0x58
#define UB954_REG_SER_ALIAS     0x5C
#define UB954_REG_SLAVE_ID      0x5D
#define UB954_REG_SLAVE_ALIAS   0x65
#define UB954_REG_PORT_DEBUG    0x4D
#define UB954_REG_CSI_CTL       0x33

struct ub954_state {
    struct i2c_client *client;
    struct regmap *regmap;
};

static const struct regmap_config ub954_regmap_config = {
    .reg_bits = 8,
    .val_bits = 8,
    .max_register = 0xFF,
};

/*
 * @brief Initialize UB954
 */
static int ub954_init(struct ub954_state *state)
{
    int ret;
    unsigned int val;

    /* 1. Software Reset */
    ret = regmap_write(state->regmap, UB954_REG_RESET, 0x01);
    if (ret < 0) return ret;
    msleep(100);

    /* 2. Check Device ID */
    ret = regmap_read(state->regmap, UB954_REG_I2C_DEV_ID, &val);
    dev_info(&state->client->dev, "Detected UB954 ID: 0x%02x\n", val);

    /* 3. Configure RX Port 0 */
    /* Select RX Port 0 for register access */
    regmap_write(state->regmap, 0x4C, 0x01); 
    
    /* Enable RX Port 0, Pass-Through, Coax Mode */
    /* Reg 0x6D: Port Config */
    /* Bit 2: Coax (1) / STP (0) */
    /* Bit 0: Port Enable */
    regmap_write(state->regmap, 0x6D, 0x7E); // Coax, Raw10, Continuous Clock

    /* 4. Configure Back Channel */
    /* Reg 0x58: BCC Config */
    /* Set I2C Pass-Through, 50Mbps BC */
    regmap_write(state->regmap, UB954_REG_BCC_CONFIG, 0x5E);

    /* 5. Configure CSI-2 Output */
    /* Reg 0x33: CSI Control */
    /* Enable 4 lanes, Continuous Clock */
    regmap_write(state->regmap, UB954_REG_CSI_CTL, 0x03);

    dev_info(&state->client->dev, "UB954 Initialized\n");
    return 0;
}
```

### Example 2: I2C Alias Configuration (The TI Way)

TI FPD-Link handles remote I2C access differently than GMSL. You must explicitly program "Alias" registers for every remote device you want to talk to.

**Concept:**
*   **SlaveID:** The physical I2C address of the remote device (e.g., Sensor at 0x10).
*   **SlaveAlias:** The address the Host uses to talk to it (e.g., 0x18).
*   **Lock:** The link must be locked before I2C works.

```c
/**
 * @brief Configure Remote I2C Aliases
 */
static int ub954_setup_aliases(struct ub954_state *state)
{
    int ret;
    
    /* Select RX Port 0 */
    regmap_write(state->regmap, 0x4C, 0x01);
    
    /* 1. Serializer Alias */
    /* Physical ID of UB953 is usually 0x18 (7-bit) or 0x30 (8-bit) */
    /* Let's say physical is 0x30 (8-bit) -> 0x18 (7-bit) */
    /* We want to talk to it at 0x18 */
    
    /* Reg 0x5B: SER ID (Physical) */
    regmap_write(state->regmap, 0x5B, 0x30); // 8-bit address
    /* Reg 0x5C: SER Alias (Host side) */
    regmap_write(state->regmap, 0x5C, 0x30); // 8-bit address
    
    /* 2. Sensor Alias */
    /* Sensor Physical: 0x20 (8-bit) -> 0x10 (7-bit) */
    /* Sensor Alias: 0x24 (8-bit) -> 0x12 (7-bit) */
    
    /* Reg 0x5D: SlaveID[0] */
    regmap_write(state->regmap, 0x5D, 0x20); // Physical
    /* Reg 0x65: SlaveAlias[0] */
    regmap_write(state->regmap, 0x65, 0x24); // Alias
    
    dev_info(&state->client->dev, "Aliases Configured: SER=0x18, Sensor=0x12\n");
    return 0;
}
```

### Example 3: GPIO Forwarding (UB953 -> UB954)

Forwarding a signal (e.g., External Frame Sync) from the Serializer side to the Deserializer side, or vice versa.

**Scenario:** ECU sends FSYNC (GPIO) to Sensor.
*   ECU GPIO -> UB954 GPIO0 -> Back Channel -> UB953 GPIO0 -> Sensor FSYNC.

```c
/**
 * @brief Configure GPIO Forwarding (Downlink: DES -> SER)
 */
static int ub954_config_gpio_fsync(struct ub954_state *state)
{
    /* 1. Configure UB954 GPIO0 as Input */
    /* Reg 0x0F: GPIO0 Config */
    /* Bit 0: Input Enable */
    regmap_write(state->regmap, 0x0F, 0x03); // GPIO0 Input, GPIO1 Input
    
    /* 2. Map UB954 GPIO0 to Back Channel GPIO0 */
    /* Reg 0x6E: BC GPIO Map */
    /* Map GPIO0 -> BC_GPIO0 */
    regmap_write(state->regmap, 0x6E, 0x00); // 0x00 means GPIO0 maps to GPIO0
    
    /* 3. Configure UB953 (Remote) GPIO0 as Output */
    /* We need to write to the Serializer Alias */
    /* This requires I2C access to the SER */
    
    /* ... (I2C write to SER Alias 0x18, Reg 0x0E) ... */
    /* Set GPIO0 as Output, Source from Back Channel */
    
    return 0;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Link Establishment & Diagnostics

**Objective:** Bring up a link between UB953 and UB954 and verify signal quality.

**Steps:**
1.  Connect UB953 EVK to UB954 EVK via Coax.
2.  Power up.
3.  Check **LOCK LED** on UB954.
4.  Read **Parity Error Count** (Reg 0x55/0x56).
    ```bash
    i2cget -y 2 0x30 0x55 # MSB
    i2cget -y 2 0x30 0x56 # LSB
    ```
    *   Ideally, this should be 0. If it's increasing, the link is unstable.

### Lab 2: Pattern Generation (PatGen)

**Objective:** Use the UB954's internal pattern generator to test the CSI-2 interface to the SoC without a camera.

**Steps:**
1.  Enable PatGen on UB954.
    *   Reg 0xB0: Indirect Access (Select Pattern Gen Block)
    *   Reg 0xB1: Pattern Config (Color Bars, Resolution)
    *   Reg 0xB2: Pattern Size
2.  Start Streaming on SoC.
    *   You should see Color Bars.
    *   This confirms the **DES -> SoC CSI-2** link is working, isolating it from the **SER -> DES** link.

### Lab 3: Cable Fault Detection

**Objective:** Simulate a cable fault (open/short) and detect it.

**Steps:**
1.  Enable Cable Fault Detection on UB954.
2.  Disconnect the cable.
3.  Read Interrupt Status Register (Reg 0x4D).
    *   Check for "Link Lost" or "Cable Fault" bits.

---

## 🐛 Debugging Techniques

### Debug 1: Adaptive EQ Failure

**Symptom:** Link fails to lock, or locks intermittently.

**Cause:** The Adaptive Equalizer (AEQ) in the UB954 cannot compensate for the cable loss (too long or poor quality).

**Debug:**
1.  Read AEQ Status (Reg 0xD2/0xD3).
    *   This tells you the EQ level chosen (0-14).
    *   If it's at 14 (Max), the signal is too weak.
2.  **Strobe Position:** Read the strobe position (Reg 0xD4). It should be centered in the eye.

**Fix:**
*   Use a shorter cable.
*   Use a higher quality cable (lower insertion loss).
*   Manually force EQ values if AEQ is unstable (advanced).

### Debug 2: I2C NACKs (Ghost Device)

**Symptom:** You try to talk to the Sensor Alias, but get NACK.

**Cause:**
1.  Link is not locked.
2.  Alias is not programmed correctly.
3.  Remote sensor is held in reset.

**Debug:**
1.  Check Lock Status (Reg 0x04, Bit 2).
2.  Verify Alias Registers (0x5D, 0x65).
3.  Check GPIO states on the Serializer (is the Sensor Reset pin low?).

---

## ⚡ Performance Optimization

### Optimization 1: CSI-2 Continuous Clock vs Non-Continuous

*   **Continuous Clock:** The CSI-2 clock lane toggles even during blanking.
    *   **Pros:** More robust, easier for some SoCs to lock.
    *   **Cons:** Higher EMI.
*   **Non-Continuous:** Clock stops during blanking (LP mode).
    *   **Pros:** Lower EMI.
    *   **Cons:** SoC receiver must support it.
*   **Config:** UB954 Reg 0x33, Bit 0.

### Optimization 2: Back Channel Frequency

*   Default is often 50 Mbps.
*   Can be reduced to 2.5 Mbps or 10 Mbps for very long cables.
*   **Trade-off:** Slower I2C access, higher latency for GPIO.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **How does TI's "Alias" based I2C access differ from Maxim's "Tunneling"?**
2.  **What is the purpose of the Adaptive Equalizer (AEQ) in the Deserializer?**
3.  **Why might you use the Internal Pattern Generator (PatGen) during bring-up?**
4.  **Explain the difference between Coax and STP modes in FPD-Link.**

### Practical Challenges

1.  **Write a script** to scan for all connected serializers on a Quad Deserializer (UB960) and assign them aliases 0x18, 0x19, 0x1A, 0x1B.
2.  **Debug a scenario** where the Link is Locked, but the Parity Error Count is incrementing by 100 every second.

---

## 📚 Further Reading & Resources

### Reference Manuals
*   **TI:** "DS90UB954-Q1 Datasheet" (The Bible for this chip).
*   **TI App Note:** "FPD-Link III SerDes Debugging Guide".

### Tools
*   **Analog LaunchPAD (ALP):** TI's GUI tool. Extremely useful for visualizing registers and link status.

---

## 🎓 Summary

Today we covered:
- ✅ **FPD-Link Architecture:** TI's approach to SerDes.
- ✅ **UB954 Driver:** Initialization and Port Configuration.
- ✅ **I2C Aliasing:** The specific mechanism for remote access in FPD-Link.
- ✅ **Diagnostics:** Parity errors, AEQ status, and Pattern Generation.
- ✅ **Comparison:** GMSL vs FPD-Link trade-offs.

**Next:** Day 18 - SerDes Link Training, Equalization & Signal Integrity.

---

**Day 17 Complete** | Phase 3: Camera Systems & ISP | Week 3: SerDes & Automotive Interfaces
