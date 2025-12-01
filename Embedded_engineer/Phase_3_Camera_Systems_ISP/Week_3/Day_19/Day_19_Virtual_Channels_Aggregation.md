# Day 19: Virtual Channels & Aggregation
## Phase 3: Camera Systems & ISP | Week 3: SerDes & Automotive Interfaces

---

## 🎯 Learning Objectives
1. **Master** MIPI CSI-2 Virtual Channels (VC) and Data Types (DT)
2. **Implement** Virtual Channel Remapping in SerDes Deserializers
3. **Configure** Quad-Camera Aggregation (merging 4 streams into 1 CSI-2 port)
4. **Distinguish** between Line-Interleaved and VC-Interleaved aggregation
5. **Calculate** bandwidth requirements for aggregated links
6. **Debug** VC collision and synchronization issues in multi-camera systems

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Quad Deserializer (MAX9286/MAX96712 or DS90UB960/964), 4x Camera Modules
*   **Software:** Linux V4L2 Subdev Framework, Media Controller API (`media-ctl`)
*   **Knowledge:** MIPI CSI-2 Packet Structure, SerDes Architecture

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Need for Aggregation

In automotive systems (Surround View, ADAS), we often have 4, 8, or even 12 cameras. Connecting each camera to a separate CSI-2 port on the SoC is impractical due to pin count and board routing complexity.

**Solution: Aggregation**
*   **Concept:** Combine video streams from multiple cameras into a single high-speed CSI-2 link.
*   **Mechanism:** The Deserializer acts as a multiplexer. It receives serial streams from 4 cameras and interleaves them onto one CSI-2 output port (e.g., 4 lanes).
*   **Separation:** The SoC separates the streams based on **Virtual Channel (VC)** IDs.

### 🔹 Part 2: Virtual Channels (VC) & Data Types (DT)

#### 2.1 MIPI CSI-2 Packet Header
Every CSI-2 packet has a 32-bit header containing:
*   **Data ID (DI):** 1 byte.
    *   **Bits 7-6:** Virtual Channel (0-3).
    *   **Bits 5-0:** Data Type (e.g., 0x2B for RAW10, 0x1E for YUV422).
*   **Word Count (WC):** 2 bytes.
*   **ECC:** 1 byte.

#### 2.2 VC-ID Remapping
Most image sensors output data on **VC0** by default.
If you connect 4 cameras (all sending VC0) to a Deserializer, and the Deserializer simply forwards them, the SoC will receive 4 interleaved streams all labeled "VC0". The SoC cannot distinguish them -> **Collision!**

**The Fix:** The Deserializer must **Remap** the VCs.
*   Cam 1 (VC0) -> Link A -> DES -> **Remap to VC0** -> SoC
*   Cam 2 (VC0) -> Link B -> DES -> **Remap to VC1** -> SoC
*   Cam 3 (VC0) -> Link C -> DES -> **Remap to VC2** -> SoC
*   Cam 4 (VC0) -> Link D -> DES -> **Remap to VC3** -> SoC

### 🔹 Part 3: Aggregation Modes

#### 3.1 Virtual Channel Interleaving (Standard)
*   **Method:** Each camera is assigned a unique VC. Packets from different cameras are interleaved on the CSI-2 bus.
*   **Pros:** Standard MIPI feature, supported by most SoCs (Jetson, Snapdragon, TDA4).
*   **Cons:** Limited to 4 cameras per CSI-2 port (since VC is 2 bits: 0-3). *Note: MIPI CSI-2 v2.0+ supports more VCs, but hardware support varies.*

#### 3.2 Line Interleaving (Data Type based)
*   **Method:** All cameras use the same VC, but lines are tagged or ordered specifically.
*   **Use Case:** FPGA processing or specific ISPs that handle "super-frames".
*   **Cons:** Complex to de-interleave in software if hardware doesn't support it.

#### 3.3 Frame Stitching (Super-Frame)
*   **Method:** The Deserializer stitches images side-by-side or top-bottom into one giant frame.
*   **Example:** 4x 1920x1080 images -> One 3840x2160 image.
*   **Pros:** Uses only 1 VC. Can support > 4 cameras.
*   **Cons:** Requires synchronized cameras. If one camera fails, the whole super-frame might be corrupted.

---

## 💻 Implementation Examples

### Example 1: Configuring VC Remapping (MAX9286 - GMSL1)

The MAX9286 is a classic Quad Deserializer. It automatically assigns VCs based on the Link ID if configured.

```c
/**
 * @file max9286_setup.c
 * @brief Configure MAX9286 for Quad Camera Aggregation
 */

#include <linux/regmap.h>

/* MAX9286 Registers */
#define REG_CSI_VC_MAP      0x15
#define REG_DEBUG_0         0x12

/*
 * @brief Configure VC Mapping
 * Maps Link 0->VC0, Link 1->VC1, Link 2->VC2, Link 3->VC3
 */
int max9286_config_vc_map(struct regmap *map)
{
    /* 
     * Register 0x15: CSI-2 Virtual Channel Mapping
     * Bits 7-6: Link 3 VC
     * Bits 5-4: Link 2 VC
     * Bits 3-2: Link 1 VC
     * Bits 1-0: Link 0 VC
     * 
     * Value 0xE4 = 11 10 01 00 (Binary)
     * Link 3 -> VC3
     * Link 2 -> VC2
     * Link 1 -> VC1
     * Link 0 -> VC0
     */
    return regmap_write(map, REG_CSI_VC_MAP, 0xE4);
}

/*
 * @brief Enable CSI-2 Output
 */
int max9286_enable_output(struct regmap *map)
{
    /* Reg 0x12: Enable CSI-2 Lanes */
    /* Enable 4 lanes */
    return regmap_write(map, REG_DEBUG_0, 0xF4); // Example value
}
```

### Example 2: Configuring VC Remapping (DS90UB960 - FPD-Link III)

The UB960 is more flexible, allowing arbitrary mapping.

```c
/**
 * @file ub960_setup.c
 * @brief Configure DS90UB960 for VC Remapping
 */

/*
 * @brief Map RX Port VCs to CSI-2 VCs
 * @param port: RX Port (0-3)
 * @param input_vc: VC coming from Sensor (usually 0)
 * @param output_vc: VC to send to SoC (0-3)
 */
int ub960_map_vc(struct regmap *map, int port, int input_vc, int output_vc)
{
    /* 1. Select RX Port */
    regmap_write(map, 0x4C, (1 << port));
    
    /* 2. Configure VC Map */
    /* Reg 0x70: VC_ID_MAP */
    /* We need to map Input VC to Output VC */
    /* This register is often complex, mapping multiple VCs */
    
    /* Simplified for UB960:
     * Reg 0x72: CSI_VC_MAP
     * Maps the incoming stream to an outgoing VC
     */
    
    unsigned int val;
    regmap_read(map, 0x72, &val);
    
    /* Clear old mapping for this input VC */
    /* Set new mapping */
    /* Note: Implementation depends on specific register bitfields */
    
    /* For UB960, it's often easier to use "CSI-2 Forwarding" 
     * where Port 0 -> VC0, Port 1 -> VC1, etc.
     */
    
    /* Reg 0x20: Forwarding Control */
    /* Enable forwarding for this port */
    
    return 0;
}
```

### Example 3: Linux V4L2 Media Controller Setup

On the SoC side (e.g., Jetson), we need to tell the VI (Video Input) driver to expect 4 VCs.

```bash
#!/bin/bash
# Configure Media Controller for Quad Camera (VC0-3)

# Reset Media Graph
media-ctl -r

# Link CSI-2 Receiver (VI) to Deserializer
# Assuming entity "max9286 2-0048" is the deserializer

# Configure Pad 0 (Output of DES)
media-ctl -V "'max9286 2-0048':0 [fmt:SRGGB10_1X10/1920x1080 field:none]"

# Configure VI Channels (Nodes /dev/video0 to /dev/video3)
# Map VC0 -> video0
# Map VC1 -> video1
# Map VC2 -> video2
# Map VC3 -> video3

# This is often handled by the device tree, but dynamic routing is possible
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Bandwidth Calculation

**Objective:** Determine if 4 cameras can fit on one CSI-2 port.

**Scenario:**
*   4x Cameras: 1920x1080 @ 30fps, Raw12.
*   CSI-2 Output: 4 Lanes, 1.5 Gbps per lane.

**Calculation:**
1.  **Per Camera Data Rate:**
    *   Pixels/sec = 1920 * 1080 * 30 * 1.2 (blanking) ≈ 75 Mpix/s.
    *   Bits/sec = 75 Mpix/s * 12 bits = 900 Mbps.
2.  **Total Input Rate:**
    *   4 * 900 Mbps = 3.6 Gbps.
3.  **Total Output Capacity:**
    *   4 Lanes * 1.5 Gbps = 6.0 Gbps.
4.  **Conclusion:** 3.6 Gbps < 6.0 Gbps. **Feasible.**
    *   *Note: Overhead (headers/footers) is small (~5-10%).*

### Lab 2: VC Collision Debugging

**Objective:** Intentionally cause a VC collision and observe the result.

**Steps:**
1.  Configure Quad Deserializer.
2.  Set Link 0 to map to VC0.
3.  Set Link 1 to *also* map to VC0 (instead of VC1).
4.  Start streaming.
5.  **Observation:**
    *   The image will likely look corrupted, tearing, or flickering.
    *   The SoC's CSI-2 error counters (CRC, ECC) will skyrocket.
    *   Frame rate might report double (60fps instead of 30fps) but with garbage data.

### Lab 3: Frame Synchronization Verification

**Objective:** Verify that all 4 cameras capture at the exact same time.

**Steps:**
1.  Point all 4 cameras at a high-speed stopwatch (e.g., on a phone screen or dedicated LED timer).
2.  Capture a frame from all 4 simultaneously.
3.  Compare the timestamp shown in the images.
4.  **Requirement:** They should be identical (within exposure time).
    *   If not, your FSYNC (Frame Sync) signal is not working or cameras are free-running.

---

## 🐛 Debugging Techniques

### Debug 1: "Frame ID" Mismatch

**Symptom:** In a surround view system, the front camera image appears in the rear camera slot occasionally.

**Cause:**
*   The SoC driver relies on packet arrival order instead of VC ID (rare but possible in simple drivers).
*   Or, the VC mapping is dynamic/unstable.

**Fix:**
*   Ensure the driver explicitly filters by VC ID.
*   Check Deserializer configuration to ensure static mapping.

### Debug 2: FIFO Overflow

**Symptom:** Frame drops or partial frames when all 4 cameras are active, but works fine with 1 camera.

**Cause:**
*   Instantaneous bandwidth exceeds CSI-2 capacity. Even if *average* bandwidth is fine, if all 4 cameras send a line *at the exact same time*, the Deserializer's internal FIFO might overflow before it can serialize to CSI-2.

**Fix:**
*   **Staggered Start:** Configure the cameras to start readout with a slight phase offset (e.g., 10 lines delay between each).
*   **Increase CSI-2 Speed:** Boost the output lane rate.

---

## ⚡ Performance Optimization

### Optimization 1: VC-X (Virtual Channel Extension)

Standard CSI-2 supports 4 VCs (2 bits).
Newer sensors and SoCs support **VC-X** (Virtual Channel Extension) or **DI-X** (Data ID Extension) to support up to 16 or 32 VCs.
*   **Use Case:** Aggregating 8 or 12 cameras.
*   **Requirement:** Hardware support on both Sensor, SerDes, and SoC.

### Optimization 2: Data Type Multiplexing

If you run out of VCs (e.g., using an older SoC), you can use Data Types to distinguish streams.
*   Cam 1 -> VC0, DT = RAW10 (0x2B)
*   Cam 2 -> VC0, DT = RAW12 (0x2C) - *If sensor supports outputting fake DT*
*   *Hack:* Some drivers use User Defined Data Types (0x30-0x37) to separate streams.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the maximum number of Virtual Channels in standard MIPI CSI-2 v1.3?**
2.  **Why is Frame Synchronization critical for aggregation?**
3.  **Explain the difference between Line Interleaved and Frame Interleaved (VC) aggregation.**
4.  **If you have 4 cameras of 2Gbps each, and a 4-lane CSI-2 output at 1.5Gbps/lane, will it work?**

### Practical Challenges

1.  **Design a register map** for a UB960 to map:
    *   Port 0 (VC0) -> VC0
    *   Port 0 (VC1) -> VC1  (Dual-stream sensor)
    *   Port 1 (VC0) -> VC2
    *   Port 1 (VC1) -> VC3
2.  **Calculate the FIFO size required** to buffer one line of 4K Raw12 video.

---

## 📚 Further Reading & Resources

### Standards
*   **MIPI Alliance:** "CSI-2 Specification" (Section on Virtual Channels).

### Datasheets
*   **Maxim:** MAX9286 Datasheet (Aggregation Mode).
*   **TI:** DS90UB960-Q1 Datasheet.

---

## 🎓 Summary

Today we covered:
- ✅ **Virtual Channels:** The key to multi-camera systems.
- ✅ **Aggregation:** Merging multiple streams in the Deserializer.
- ✅ **VC Remapping:** Preventing ID collisions.
- ✅ **Bandwidth:** Calculating limits for aggregated links.
- ✅ **Synchronization:** Ensuring coherent capture.

**Next:** Day 20 - Automotive Camera Synchronization over SerDes (FSYNC).

---

**Day 19 Complete** | Phase 3: Camera Systems & ISP | Week 3: SerDes & Automotive Interfaces
