# Day 20: Automotive Camera Synchronization (FSYNC)
## Phase 3: Camera Systems & ISP | Week 3: SerDes & Automotive Interfaces

---

## 🎯 Learning Objectives
1. **Understand** the critical need for microsecond-level synchronization in ADAS/Autonomous Driving
2. **Implement** Frame Sync (FSYNC) generation using SerDes GPIO forwarding
3. **Configure** Master/Slave synchronization topologies
4. **Analyze** synchronization latency and jitter over GMSL/FPD-Link
5. **Integrate** camera timestamps with LiDAR/Radar using PTP (gPTP 802.1AS)
6. **Debug** synchronization failures (rolling shutter artifacts, phase mismatch)

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Quad SerDes System (MAX9286/UB960), 4x Global Shutter Cameras (preferred) or Rolling Shutter
*   **Software:** Oscilloscope, Linux PTP daemon (`linuxptp`)
*   **Knowledge:** GPIO Forwarding (Day 16/17), Rolling Shutter vs Global Shutter

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Sync Matters?

In a moving vehicle (e.g., 100 km/h = 27.7 m/s), a timing error of 10ms corresponds to 27cm of motion.
*   **Stereo Vision:** Left and Right images must be captured at the *exact* same instant. Any delay causes depth estimation errors.
*   **Sensor Fusion:** Camera, LiDAR, and Radar data must be fused. If Camera is 30ms late, the object position will be wrong.
*   **Surround View:** Misaligned frames cause "ghosting" or "tearing" at the stitch lines.

**Requirement:** Synchronization accuracy < 100 microseconds (often < 10 us).

### 🔹 Part 2: Synchronization Methods

#### 2.1 Software Trigger (Bad)
*   Host sends I2C command "Start" to all cameras sequentially.
*   **Problem:** I2C is slow (400kHz). There is a delay between Cam 1 and Cam 4 (ms range). OS scheduling adds jitter.
*   **Verdict:** Unusable for ADAS.

#### 2.2 Hardware Trigger (FSYNC) - The Standard
*   **Mechanism:** A dedicated GPIO pin on the image sensor triggers the exposure.
*   **Topology:**
    *   **External Master:** An FPGA or MCU generates a pulse -> Deserializer GPIO -> Back Channel -> Serializer GPIO -> Sensor FSYNC.
    *   **Deserializer Master:** The Deserializer itself generates the pulse internally.

#### 2.3 PTP (Precision Time Protocol)
*   **Mechanism:** Network-based time synchronization (IEEE 802.1AS).
*   **Usage:** Synchronizes the *System Time* of the Host, LiDAR, and Radar. Cameras are usually "slaves" to this time via the FSYNC pulse which is timestamped by the SoC.

### 🔹 Part 3: SerDes FSYNC Architecture

**Path:**
1.  **Generator:** Deserializer internal timer OR External GPIO Input.
2.  **Distribution:** Deserializer forwards this signal to ALL Back Channels simultaneously.
3.  **Local Output:** Serializer outputs signal on GPIO.
4.  **Sensor:** Sensor starts exposure on rising edge.

**Latency:**
*   GMSL/FPD-Link Back Channel latency is deterministic and very low (< 2us).
*   Since the signal travels to all cameras over similar cable lengths, the *relative* skew is negligible (< 100ns).

---

## 💻 Implementation Examples

### Example 1: Configuring Internal FSYNC Generation (MAX9286)

The MAX9286 has a built-in frame sync generator.

```c
/**
 * @file max9286_fsync.c
 * @brief Configure Internal FSYNC Generation
 */

#include <linux/regmap.h>

/* MAX9286 Registers */
#define REG_FSYNC_PERIOD_L  0x06
#define REG_FSYNC_PERIOD_M  0x07
#define REG_FSYNC_PERIOD_H  0x08
#define REG_FSYNC_CONFIG    0x01

/*
 * @brief Configure FSYNC Generator
 * @param fps: Target Frame Rate (e.g., 30)
 * @param pclk: Pixel Clock in Hz (needed for calculation, or internal osc)
 * Note: MAX9286 uses PCLK cycles or Internal Osc for timing.
 */
int max9286_enable_fsync(struct regmap *map, int fps)
{
    /* 
     * Formula: Period = PCLK / FPS
     * Simplified: We program the number of cycles.
     * Let's assume we use the manual mode or internal timer.
     */
    
    /* 1. Set FSYNC Mode to "Internal Manual" or "Automatic" */
    /* Reg 0x01: Bit 1-0: 10 = Automatic Internal */
    regmap_write(map, REG_FSYNC_CONFIG, 0x02);
    
    /* 2. Program Period (Example for 30fps) */
    /* Value depends on clock source. Let's assume 27MHz ref. */
    /* 27,000,000 / 30 = 900,000 cycles */
    uint32_t period = 900000;
    
    regmap_write(map, REG_FSYNC_PERIOD_L, period & 0xFF);
    regmap_write(map, REG_FSYNC_PERIOD_M, (period >> 8) & 0xFF);
    regmap_write(map, REG_FSYNC_PERIOD_H, (period >> 16) & 0xFF);
    
    /* 3. Enable FSYNC Output on GPIO */
    /* Configure GPIO0/1 as FSYNC outputs */
    
    return 0;
}
```

### Example 2: External FSYNC Forwarding (UB960)

The UB960 takes an external pulse (from ECU/FPGA) and broadcasts it.

```c
/**
 * @brief Configure UB960 for External FSYNC
 * GPIO0 Input -> BC_GPIO0 -> All Ports
 */
int ub960_config_ext_fsync(struct regmap *map)
{
    /* 1. Configure GPIO 0 as Input */
    /* Reg 0x0F: GPIO0 Config */
    regmap_write(map, 0x0F, 0x03); // Input, Enable
    
    /* 2. Map GPIO 0 to Back Channel GPIO 0 */
    /* Reg 0x6E: BC GPIO Map */
    regmap_write(map, 0x6E, 0x00); // GPIO0 -> BC_GPIO0
    
    /* 3. Enable Forwarding on all Ports */
    /* Reg 0x32: Port Forward Control */
    /* Enable forwarding to Port 0, 1, 2, 3 */
    regmap_write(map, 0x32, 0x0F); // Broadcast
    
    /* 
     * Note: You also need to configure the Serializers (UB953)
     * to output BC_GPIO0 to their local GPIO pin connected to the Sensor.
     * This is done via I2C aliasing (Day 17).
     */
    
    return 0;
}
```

### Example 3: Timestamping in Driver (V4L2)

The SoC receives the frame. It's crucial to timestamp it accurately.
*   **SOF (Start of Frame):** The moment the first packet arrives.
*   **Hardware Timestamping:** The CSI-2 receiver hardware captures the system timer value at SOF.

```c
/**
 * @brief V4L2 Buffer Timestamping
 * In the CSI-2 receiver driver interrupt handler
 */
static irqreturn_t tegra_vi_isr(int irq, void *data)
{
    struct tegra_channel *chan = data;
    struct timespec64 ts;
    
    /* Read Hardware Timestamp Register */
    /* This captures the time when the Frame Start code was received */
    u64 hw_time = readl(chan->vi_base + VI_TIMESTAMP_REG);
    
    /* Convert to System Time (Monotonic) */
    /* Often handled by kernel core */
    ktime_get_ts64(&ts);
    
    /* Fill V4L2 Buffer Timestamp */
    struct vb2_v4l2_buffer *vbuf = next_buffer(chan);
    vbuf->vb2_buf.timestamp = hw_time; // Nanoseconds
    
    /* ... */
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Oscilloscope Verification

**Objective:** Measure the skew between 4 cameras.

**Steps:**
1.  Probe the FSYNC pin on Camera 1 (Serializer GPIO).
2.  Probe the FSYNC pin on Camera 2.
3.  Trigger on Cam 1 Rising Edge.
4.  Measure delay to Cam 2 Rising Edge.
5.  **Result:** Should be < 1 microsecond. If it's milliseconds, you are not using hardware sync!

### Lab 2: Rolling Shutter Effect

**Objective:** Demonstrate why sync matters for moving objects.

**Steps:**
1.  Set up 2 cameras in stereo configuration.
2.  Disable FSYNC (let them free-run).
3.  Move a vertical bar (or wave your hand) quickly across the field of view.
4.  Capture an image pair.
5.  **Observation:** The bar will be at different positions in Left and Right images, and might be slanted (Rolling Shutter effect).
6.  **Enable FSYNC:** Repeat. The bar should be at the correct disparity-shifted position, with identical slant.

### Lab 3: PTP Synchronization (Linux)

**Objective:** Sync the Jetson/ECU clock to a Master Clock.

**Steps:**
1.  Connect Jetson to a PTP Master (e.g., another PC or Switch) via Ethernet.
2.  Run `ptp4l` (PTP for Linux).
    ```bash
    sudo ptp4l -i eth0 -m -S
    ```
3.  Observe the offset. It should converge to < 1000ns.
4.  Now, all camera timestamps (derived from system time) are globally synchronized.

---

## 🐛 Debugging Techniques

### Debug 1: Phase Drift

**Symptom:** Cameras start synchronized but slowly drift apart over minutes.

**Cause:**
*   Using "Software Trigger" or free-running mode with slightly different crystal frequencies.
*   FSYNC signal is intermittent.

**Fix:**
*   Ensure FSYNC is continuous.
*   Verify the Sensor is in "Slave Mode" (External Trigger Mode). If it's in Master Mode, it ignores the FSYNC pulse!

### Debug 2: "Tearing" in Surround View

**Symptom:** The stitch line on the ground moves or tears.

**Cause:**
*   Latency difference between cameras.
*   One camera is 1 frame behind the others (FIFO buffering issue).

**Fix:**
*   Check the Frame Counter embedded in the image data (if sensor supports it).
*   Ensure all cameras are triggered by the *same* pulse edge.

---

## ⚡ Performance Optimization

### Optimization 1: Trigger-to-Exposure Latency

Sensors have a delay between receiving the FSYNC pulse and actually starting exposure.
*   **Optimization:** Characterize this delay (datasheet).
*   **Compensation:** If mixing different sensors (e.g., 2x IMX390 and 2x AR0231), their delays might differ. You might need to delay the FSYNC pulse to the faster sensor using the Serializer's GPIO delay feature.

### Optimization 2: Duty Cycle

*   FSYNC is usually a pulse.
*   Some sensors require a specific duty cycle or pulse width.
*   **Config:** Adjust the GPIO output pulse width on the Serializer/Deserializer to meet sensor specs (e.g., > 100us).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is I2C-based synchronization insufficient for autonomous driving?**
2.  **Explain the path of an FSYNC pulse in a GMSL system.**
3.  **What is the difference between Global Shutter and Rolling Shutter regarding synchronization?**
4.  **How does PTP help in multi-sensor fusion?**

### Practical Challenges

1.  **Calculate the motion blur** of a car moving at 60mph if the exposure time is 10ms.
2.  **Design a sync scheme** for a system with 4 Cameras (30fps) and 1 LiDAR (10Hz). How do you align them?

---

## 📚 Further Reading & Resources

### Standards
*   **IEEE 802.1AS:** Timing and Synchronization for Time-Sensitive Applications in Bridged Local Area Networks (gPTP).

### App Notes
*   **Sony:** "IMX Sensor Synchronization Application Note".
*   **Maxim/TI:** "GPIO Forwarding for Frame Sync".

---

## 🎓 Summary

Today we covered:
- ✅ **FSYNC:** The heartbeat of the camera system.
- ✅ **Hardware Sync:** Using SerDes GPIOs for microsecond accuracy.
- ✅ **Topologies:** Internal vs External generation.
- ✅ **PTP:** Global time synchronization.
- ✅ **Debugging:** Solving drift and phase issues.

**Next:** Day 21 - Week 3 Review & SerDes Integration Project.

---

**Day 20 Complete** | Phase 3: Camera Systems & ISP | Week 3: SerDes & Automotive Interfaces
