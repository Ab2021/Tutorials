# Day 74: Phase 3 Capstone Project - Design & Architecture
## Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project

---

## 🎯 Learning Objectives
1.  **Define** the scope of the Capstone Project: "The Intelligent Traffic Camera".
2.  **Gather** Requirements: 4K Resolution, License Plate Recognition (LPR), Night Vision, LTE Streaming.
3.  **Design** the System Architecture: Sensor -> ISP -> AI -> Encoder -> Cloud.
4.  **Select** Components: IMX415 (4K Starvis), Jetson Nano/Orin, M12 Lens.
5.  **Draft** the Software Stack: V4L2 Driver, GStreamer Pipeline, DeepStream Inference.

---

## 📚 Prerequisites & Preparation
*   **Goal:** Apply *everything* learned in Phase 3 (Sensors, ISP, AI, Streaming, Safety).
*   **Deliverable:** A Design Document (Architecture Diagram, BOM, API Spec).

---

## 📖 Project Brief: "The Intelligent Traffic Camera"

### 🔹 Problem Statement
Design a smart camera system for a toll booth or parking lot that can:
1.  Capture high-resolution images of vehicles day and night.
2.  Automatically read License Plates (ANPR/LPR).
3.  Stream live video to a remote control center.
4.  Operate reliably outdoors (Heat, Rain, Power Loss).

### 🔹 Requirements

#### Functional
*   **Resolution:** 4K (3840x2160) to read plates at 20m.
*   **Low Light:** Must work at 0.1 Lux (Street lighting).
*   **AI:** LPR accuracy > 95%.
*   **Connectivity:** RTSP Stream + MQTT Metadata.

#### Non-Functional
*   **Latency:** < 200ms for live view.
*   **Boot Time:** < 5s.
*   **Power:** < 10W.
*   **Reliability:** Watchdog protected, Thermal throttling.

---

## 🏗️ System Architecture

### 1. Hardware Block Diagram

```mermaid
graph TD
    LENS[M12 Lens f/1.6] --> SENSOR[Sony IMX415]
    SENSOR -->|MIPI CSI-2 (4 Lanes)| SOC[NVIDIA Jetson Orin Nano]
    
    SOC -->|ISP| MEM[DRAM]
    MEM -->|GPU| AI[AI Inference]
    MEM -->|ENC| H264[H.264 Encoder]
    
    H264 -->|Network| LTE[4G/5G Modem]
    AI -->|Metadata| LTE
    
    PMIC[Power Management] --> SOC
    THERM[Temp Sensor] --> SOC
```

### 2. Software Stack

*   **OS:** Linux (Ubuntu / Tegra).
*   **Driver:** V4L2 Subdev (IMX415).
*   **Middleware:** GStreamer / DeepStream.
*   **AI Model:** LPRNet (NVIDIA TAO Toolkit).
*   **App:** C++ Daemon (`traffic_cam_service`).

---

## 💻 Implementation Plan

### Step 1: Component Selection

*   **Sensor:** Sony IMX415.
    *   *Why?* 4K Resolution, Starvis (Great Low Light), Reasonable Cost.
*   **Lens:** 6mm or 8mm focal length.
    *   *Why?* Narrow FOV to zoom in on plates at distance.
*   **SoC:** Jetson Orin Nano.
    *   *Why?* Powerful GPU for AI, Hardware Encoder, ISP support.

### Step 2: Pipeline Design (GStreamer)

```bash
# Conceptual Pipeline
nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM), width=3840, height=2160, format=NV12' ! \
  tee name=t \
  t. ! queue ! nvstreammux ! nvinfer (LPR) ! nvdsosd ! nvvideoconvert ! nvv4l2h264enc ! rtspclientsink \
  t. ! queue ! nvjpegenc ! multifilesink (Snapshots)
```

### Step 3: Interface Definition (API)

The camera exposes a REST API for configuration.

```json
POST /api/v1/config
{
  "resolution": "4K",
  "fps": 30,
  "roi": [100, 100, 1000, 500],
  "night_mode": "auto"
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: BOM Calculation

**Objective:** Cost estimation.

*   IMX415 Module: $40
*   Jetson Orin Nano: $299
*   Carrier Board: $100
*   Lens: $20
*   Housing/Cabling: $50
*   **Total:** ~$510.

### Lab 2: FOV Calculation

**Objective:** Verify Lens selection.

*   Sensor Size (IMX415): 1/2.8" (Diagonal 6.4mm). Width ~5.6mm.
*   Distance ($D$): 20m.
*   Car Width ($W$): 2m.
*   Focal Length ($f$) = $(SensorWidth \times D) / FieldWidth$.
*   If we want to see 5m width at 20m: $f = (5.6 \times 20) / 5 = 22.4mm$.
*   *Correction:* 8mm lens gives ~14m width at 20m. Good for seeing multiple lanes.

### Lab 3: Bandwidth Estimation

**Objective:** 4G Data Plan.

*   4K H.264 Stream @ 8Mbps.
*   24 hours/day = $8 \text{ Mbps} \times 3600 \times 24 / 8 = 86 \text{ GB/day}$.
*   *Too high!*
*   **Optimization:** Only stream low-res (720p) for preview. Record 4K locally. Upload 4K clips only on "Event" (Plate Detected).

---

## 🐛 Design Risks & Mitigation

### Risk 1: Overheating in Sun

*   **Mitigation:** White enclosure (reflects sun), Active Fan, Thermal Throttling logic (Day 73).

### Risk 2: Motion Blur at Night

*   **Mitigation:** High Gain + Short Exposure. (Noise is better than Blur for LPR). IR Illuminator (850nm).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why use a "Global Shutter" sensor for traffic?** (To avoid rolling shutter distortion on fast moving cars. *Note: IMX415 is Rolling, so we must use short exposure*).
2.  **What is "WDR" (Wide Dynamic Range)?** (Ability to see plate (bright) and car (dark) simultaneously).
3.  **Why MQTT for metadata?** (Lightweight, works on bad networks).

### Practical Challenges

1.  **Draft the `device_tree` node:** Write the DTS snippet for the IMX415 connected to CSI-Port A.
2.  **Design the Database:** Schema for storing `PlateNumber`, `Timestamp`, `ImageURL`, `Confidence`.

---

## 📚 Further Reading & Resources

### Documentation
*   **NVIDIA DeepStream SDK Guide.**
*   **Sony IMX415 Datasheet.**

---

## 🎓 Summary

Today we covered:
- ✅ **Scope:** Intelligent Traffic Camera.
- ✅ **Hardware:** Sensor, Lens, SoC selection.
- ✅ **Architecture:** Pipeline design.
- ✅ **Math:** FOV and Bandwidth.
- ✅ **Risks:** Thermal and Optical.

**Next:** Day 75 - Phase 3 Capstone Project: Implementation & Demo.

---

**Day 74 Complete** | Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project
