# Day 90: Phase 3 Final Assessment & Graduation
## Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career

---

## 🎓 Congratulations!
You have reached the end of **Phase 3: Camera Systems, SerDes & ISP Development**.
Over the last 90 days, you have transformed from an Embedded Engineer into a **Camera Systems Expert**.

---

## 📝 Final Assessment (The "Bar Exam" for Camera Engineers)

**Instructions:**
*   Time Limit: 3 Hours.
*   Passing Score: 80%.
*   Tools: Datasheets allowed. No Google.

### Section 1: Fundamentals (Physics & Hardware)
1.  **Calculate the Focal Length** needed to see a 3m wide lane at 50m distance using a 1/2" sensor (6.4mm width).
2.  **Explain "Chief Ray Angle" (CRA).** Why must the Lens CRA match the Sensor CRA?
3.  **Draw the MIPI CSI-2 Physical Layer.** Label LP-11, LP-01, HS-Zero, HS-Sync.
4.  **What is the difference between GMSL2 and FPD-Link III?** (Encoding, Bandwidth, Back-channel).
5.  **Why do we use "AC Coupling" capacitors on SerDes links?**

### Section 2: ISP & Image Quality
6.  **Describe the "Bayer Demosaic" process.** What artifacts can it cause (Zipper, False Color)?
7.  **How does "High Dynamic Range" (HDR) work?** Explain DOL (Digital Overlap) vs ME (Multi-Exposure).
8.  **What is "Lens Shading Correction" (LSC)?** How do you calibrate it?
9.  **Define "MTF50".** Why is it better than "TV Lines"?
10. **What is $\Delta E_{2000}$?** What is an acceptable value for a Macbeth Chart?

### Section 3: Software & Drivers
11. **Write the V4L2 ioctl sequence** to capture one frame (Open -> ... -> StreamOn -> ... -> Close).
12. **What is a "Sub-device" in the Linux Media Controller framework?**
13. **Explain "Zero-Copy" in GStreamer.** How does `dmabuf` work?
14. **How do you debug a "Green Screen" issue?** (UV plane zeroed? YUV format mismatch?).
15. **What is the role of `v4l2-ctl`?** Give 3 examples of its usage.

### Section 4: Advanced Topics (AI, 3D, Automotive)
16. **How does "Stereo Vision" calculate depth?** Formula: $Z = (f \times B) / d$.
17. **What is "TensorRT"?** Why is INT8 faster than FP32?
18. **Explain "ASIL B" requirements for a camera.** (CRC, Watchdog, Frame Counter).
19. **What is "Extrinsic Calibration" in Sensor Fusion?**
20. **How do you secure a camera against "Root Access" via UART?**

---

## 🏆 The Masterpiece Project: "Autonomous Delivery Robot Vision"

**Scenario:**
You are the Lead Camera Architect for a sidewalk delivery robot startup.

**Requirements:**
1.  **Front Camera:** 4K, HDR, 120deg FOV. Detects Pedestrians/Traffic Lights.
2.  **Stereo Camera:** Depth sensing for obstacle avoidance (0.5m to 5m).
3.  **Rear Camera:** 1080p, Fisheye. Reversing aid.
4.  **Compute:** Jetson Orin NX.
5.  **Connectivity:** LTE Cloud Streaming (Low Latency).

**Deliverables:**
1.  **Architecture Diagram:** Sensors -> Deserializers -> CSI Ports -> ISP -> AI -> Cloud.
2.  **BOM (Bill of Materials):** Select specific Sensors (IMX...), SerDes chips, and Lenses.
3.  **Power Budget:** Estimate total power (Cameras + SerDes + Compute).
4.  **Software Stack:** Define the GStreamer pipeline and ROS 2 nodes.
5.  **Validation Plan:** How will you test "Sun Glare" and "Rain"?

---

## 🚀 Transition to Phase 4

**Phase 3** focused on the **"Eye"** (The Camera).
**Phase 4** will focus on the **"Brain"** (The OS and Kernel).

**Phase 4: Embedded Linux & Kernel Development**
*   **Week 16-20:** Yocto Project & Buildroot (Building your own Distro).
*   **Week 21-25:** Linux Kernel Internals (Scheduler, Memory Management).
*   **Week 26-30:** Writing Complex Device Drivers (PCIe, DMA, Block).
*   **Week 31-35:** Debugging & Tracing (Ftrace, Perf, Crash).

**Preparation for Day 91:**
*   Install **Ubuntu 22.04 LTS** (Dual Boot or Dedicated Machine).
*   Buy a **BeagleBone Black** or **Raspberry Pi 4** (for Kernel hacking).
*   Get ready to compile the Linux Kernel from source!

---

## 📚 Final Words

> "A camera is a device that teaches us how to see without a camera." - Dorothea Lange

You now possess the rare skill set to build the eyes of the machines that will shape our future. Whether it's a self-driving car, a surgical robot, or a smart doorbell, you can build it.

**Good luck, and see you in Phase 4!**

---

**Day 90 Complete** | Phase 3: Camera Systems & ISP | Week 15: Final Capstone & Career
