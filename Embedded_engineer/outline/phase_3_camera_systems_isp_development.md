# 📍 Phase 3 — Camera Systems, SerDes & ISP Engineering
### Duration: **Day 261 → Day 350**
### Lab Format: **Guided (Step-by-Step Hands-On)**

> This phase transitions from kernel‑level device driver proficiency (Phase 2) into full camera system engineering including CSI‑2, SerDes, ISP tuning, HDR pipelines, metadata handling, and multi‑camera synchronization.
> All work is performed on real hardware when possible.

---

## 📌 Section 3.1 — MIPI CSI-2 Fundamentals (Days 261–280)
> Reference sources integrated from fileciteturn0file1 (MIPI CSI‑2 + V4L2 pipeline) and fileciteturn0file0 (Camera Hardware & Serializer Topics).

---

### **Week 1 (Days 261–267): CSI‑2 Signaling, PHY, and Packet Protocol**

| Day | Topic | Learning Outcome | Required Study | Guided Lab |
|-----|--------|------------------|----------------|------------|
| **Day 261** | Introduction to CSI‑2 Architecture | Understand CSI‑2 layered stack: PHY → Link → Protocol → App | MIPI CSI‑2 v3.0 Overview | Identify CSI lane count, clock lane, and expected bandwidth from sensor datasheet |
| **Day 262** | CSI‑2 PHY Modes (D‑PHY / C‑PHY) | Distinguish signaling style, lane pairing, LP/HS modes | Vendor PHY Timing Tables | Measure LP↔HS transitions using logic analyzer / oscilloscope if hardware available |
| **Day 263** | Lane Muxing & Virtual Channels | VC, DT (data type), stream splitting, multi‑camera muxing concept | Media Controller Topology Docs | Print media graph links using `media-ctl -p` and annotate VC assignments |
| **Day 264** | CSI‑2 Packet Format | Parse short packets, long packets, ECC/CRC, frame boundaries | MIPI Protocol Docs | Capture raw packet dump using SoC debug node (if supported) and decode headers |
| **Day 265** | Timing Parameters | t‑HS‑PREPARE, t‑INIT, ULPS, escape mode behavior | CSI PHY Register Mapping | Modify sensor driver timing registers and compare trace logs |
| **Day 266** | Debugging CSI‑2 Issues | Interpreting common failures: no‑probe, no‑signal, corrupted HDR metadata, lane skew | Kernel CSI debug frameworks | Force misconfigured timings and recover using driver patches |
| **Day 267** | Weekly Review & Report | Consolidate conceptual and hands‑on experience | — | Submit CSI‑2 bring‑up validation table |

---

### **Week 2 (Days 268–274): CSI‑2 Integration with V4L2 & Subdevice Framework**

| Day | Topic | Outcome | Reading | Guided Lab |
|-----|-------|--------|---------|------------|
| **Day 268** | V4L2 + Subdevice Model w/ CSI Integration | Understand sensor → CSI → ISP → memory mapping | V4L2 Kernel Docs | Visualize full pad/entity pipeline via `media-ctl` diagrams |
| **Day 269** | Endpoint Definition in Device Tree | `ports`, `endpoint`, `remote‑endpoint` link graph creation | Device Tree + Linux Media Docs | Create working endpoint entry on test sensor node |
| **Day 270** | PHY Configuration via Driver | Setting lane count, clock frequency, LP/HS mode flags | CSI Receiver Driver Code | Modify and validate lane configuration change |
| **Day 271** | Pixel Formats: RAW8/10/12/14, Bayer Orders | Correctly map output formats from sensor | Kernel Pixel Format Table | Capture images at two pixel depths and validate via `v4l2-ctl --stream-mmap` |
| **Day 272** | CSI Receiver Internal Registers | Bring‑up flow: sensor → CSI sync → streaming enable | Vendor TRM Register Maps | Read CSI receiver status and validate link result |
| **Day 273** | Timestamp + Frame Numbering | Ensure deterministic frame metadata for ADAS stacks | Request API Documentation | Enable per‑frame sequence numbering in driver |
| **Day 274** | Week 2 Checkpoint | — | — | Submit working CSI pipeline screenshot + dmesg logs |

---

### **Week 3 (Days 275–280): Error Handling, Signal Integrity & Validation**

| Day | Topic | Outcome | Guided Lab |
|-----|-------|---------|------------|
| **Day 275** | ECC/CRC Recovery Mechanisms | Detect & classify link corruption events | Force induced packet error and capture kernel logs |
| **Day 276** | Lane Skew & Signal Integrity | Identify skew tolerance, mismatch alignment | Test streaming under altered clock settings |
| **Day 277** | ULPS & Low‑Power Behavior | Implement/test entry & exit sequences | Trigger standby events and verify fast recovery |
| **Day 278** | High‑Temp/Low‑Temp Stress Patterns | Understand automotive temperature robustness expectations | Run endurance stream >30 min and validate error rate |
| **Day 279** | Multi‑Sensor CSI Switch/Multiplex Scenarios | Handle virtual channel switching under load | Enable dual‑sensor VC routing and verify link integrity |
| **Day 280** | Section Milestone Evaluation | Validate CSI‑2 full bring‑up + system stability | Final report + validated capture clips |

---

🎯 **End of Section 3.1 Expected Output:**
- Working CSI‑2 camera streaming RAW at stable FPS
- Documented DT bindings
- Debug logs for packet parsing, timing tuning, error classification
- Verified timestamp + metadata accuracy

---

➡ Next: **Section 3.2 — Sensor Frameworks, CCS, and RAW Processing Pipeline (Days 281–300)** will be added in the next update.

