# Day 81: Production Line Testing (EOL - End of Line)
## Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Design** an End-of-Line (EOL) Tester for mass production.
2.  **Implement** the Test Sequence: Electrical -> Optical -> Calibration -> OTP.
3.  **Detect** Blemishes (Dead Pixels, Dust) automatically.
4.  **Program** the OTP (One Time Programmable) memory with Module Info and Calibration Data.
5.  **Optimize** Cycle Time (UPH - Units Per Hour).
6.  **Analyze** Yield and Pareto Charts for failure modes.

---

## 📚 Prerequisites & Preparation
*   **Context:** You have a Golden Sample. Now make 1 million copies.
*   **Hardware:** Test Fixture (Pogo pins), Light Box (Uniform), Chart.
*   **Software:** Test Sequencer (Python/LabVIEW/TestStand).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The EOL Station
*   **Fixture:** Holds the camera module precisely. Uses Pogo Pins to contact the connector pads (if not using a real connector).
*   **Light Box:** Provides uniform D65 light.
*   **Chart:** Usually a "Multi-Chart" containing:
    *   Slanted Edges (Focus).
    *   Gray Patch (Color/Noise).
    *   QR Code (Traceability).

### 🔹 Part 2: The Test Sequence
1.  **Open/Short:** Check for electrical shorts on power rails.
2.  **Current:** Measure IDD (Standby and Active).
3.  **I2C:** Verify Device ID (Who am I?).
4.  **Stream:** Start video. Check for frame drops.
5.  **Focus:** Measure MTF at center and corners. (Adjust lens if Active Alignment).
6.  **Color/LSC:** Measure shading and WB.
7.  **Blemish:** Find dust/dead pixels.
8.  **OTP:** Write calibration data.
9.  **Final Check:** Verify OTP readback.

### 🔹 Part 3: OTP (One Time Programmable) Memory
*   Sensors have a small non-volatile memory (e.g., 8KB).
*   **Content:**
    *   **Module Info:** Serial Number, Date, Lens ID.
    *   **LSC Data:** The Lens Shading mesh specific to *this* unit.
    *   **AWB Data:** The R/G and B/G ratios of *this* unit under D65.
    *   **Defect List:** Coordinates of bad pixels.

---

## 💻 Implementation Examples

### Example 1: Blemish Detection Algorithm

Finding dust spots.

```python
import cv2
import numpy as np

def find_blemishes(image):
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur to estimate background (Low Pass)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    
    # 3. Subtract Background (High Pass)
    diff = cv2.absdiff(gray, bg)
    
    # 4. Threshold
    _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    
    # 5. Count Blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5: # Ignore single pixel noise
            print(f"Blemish found at {cv2.boundingRect(c)}")
            return False # Fail
            
    return True # Pass
```

### Example 2: OTP Structure (Struct)

Defining the memory map.

```c
struct otp_map {
    uint8_t header[4];       // "CAM1"
    uint32_t serial_num;     // Unique ID
    uint16_t production_date;// YYYYMMDD encoded
    uint16_t awb_rg_ratio;   // R/G * 1024
    uint16_t awb_bg_ratio;   // B/G * 1024
    uint8_t lsc_grid[221];   // 17x13 mesh
    uint8_t checksum;        // CRC8
};
```

### Example 3: Test Sequencer (Python)

```python
def run_eol_test(camera):
    try:
        # 1. Electrical
        current = psu.measure_current()
        if current > 0.3: raise Fail("Overcurrent")
        
        # 2. Functional
        camera.start()
        img = camera.capture()
        
        # 3. Optical
        mtf = analyze_mtf(img)
        if mtf < 0.4: raise Fail("Focus Bad")
        
        blemish = find_blemishes(img)
        if not blemish: raise Fail("Dust Detected")
        
        # 4. Calibration
        cal_data = calculate_calibration(img)
        camera.write_otp(cal_data)
        
        print("PASS")
        return True
        
    except Fail as e:
        print(f"FAIL: {e}")
        return False
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Simulating OTP Write

**Objective:** Store data in the sensor.

**Steps:**
1.  Most sensors (IMX219) don't have user-writable OTP exposed easily, or it's already written.
2.  **Simulation:** Use a file `otp.bin` on the SD card to represent the OTP.
3.  **Task:** Write a script that generates a unique Serial Number and "burns" it to the file.
4.  **Driver:** Modify the Camera Driver to read this file on probe and print "Camera Serial: XYZ".

### Lab 2: Defect Pixel Mapping

**Objective:** Create the Defect List.

**Steps:**
1.  Capture a "Dark Frame" (Lens covered, High Gain).
2.  Find pixels > Threshold (Hot Pixels).
3.  Capture a "Flat Field" (White wall).
4.  Find pixels < Threshold (Dead Pixels).
5.  **Output:** A list of (X, Y) coordinates.
6.  **ISP:** Feed this list to the DPC (Defect Pixel Correction) block to interpolate them out.

### Lab 3: Cycle Time Optimization

**Objective:** Go fast.

**Steps:**
1.  Measure the time for each step in Example 3.
2.  **Bottleneck:** Usually "Capture" (waiting for AE to settle) or "MTF Analysis" (CPU).
3.  **Fix:**
    *   Use fixed Exposure/Gain (skip AE settling).
    *   Use GPU for MTF.
    *   Pipeline: Capture Image N while Analyzing Image N-1.

---

## 🐛 Debugging Production Issues

### Debug 1: Yield Drop

**Symptom:** 20% of cameras failing Focus.

**Cause:**
*   Fixture misalignment. The camera is tilted relative to the chart.
*   Lens glue curing process changed.
*   **Fix:** Recalibrate the test station (Golden Sample check).

### Debug 2: OTP Checksum Error

**Symptom:** Driver refuses to load calibration.

**Cause:**
*   I2C noise during writing.
*   Power cut during writing.
*   **Fix:** Verify CRC immediately after writing. If fail, retry (if OTP allows, usually it has limited write cycles or "Page" structure).

---

## ⚡ Performance Optimization

### Optimization 1: Parallelism (Gang Testing)

*   Test 4 cameras at once.
*   Use a 4-up frame grabber.
*   Quadruples UPH (Units Per Hour).

### Optimization 2: "Golden Unit" Monitoring

*   Run a known good camera every shift (morning/evening).
*   If the Golden Unit fails, the *Tester* is broken, not the cameras.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Active Alignment"?** (Adjusting the lens position *while* looking at the image to maximize MTF, then gluing it).
2.  **Why do we need a "Dark Frame" test?** (To find hot pixels that look like stars in night video).
3.  **What is "UPH"?** (Units Per Hour. Key metric for factory cost).
4.  **Why is OTP "One Time"?** (Uses fuses or anti-fuses. Once blown, cannot be reverted. Secure).

### Practical Challenges

1.  **Design the Label:** Create a 2D Data Matrix code containing the Serial Number and Part Number.
2.  **Write a "Yield Monitor":** A script that parses the log files and plots a pie chart of Failure Modes (e.g., 5% Focus, 2% Dust, 1% Electrical).

---

## 📚 Further Reading & Resources

### Standards
*   **EMVA 1288 (again).**

---

## 🎓 Summary

Today we covered:
- ✅ **EOL:** The final gate.
- ✅ **Sequence:** Elec -> Opt -> Cal -> OTP.
- ✅ **Blemish:** Finding dust.
- ✅ **OTP:** The camera's identity.
- ✅ **Yield:** Tracking success.

**Next:** Day 82 - Week 14 Review & Compliance Checklist.

---

**Day 81 Complete** | Phase 3: Camera Systems & ISP | Week 14: Testing, Validation & Compliance
