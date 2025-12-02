# Day 73: Thermal Management & Reliability
## Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project

---

## 🎯 Learning Objectives
1.  **Understand** the Thermal Challenges: Sensors get noisy when hot; ISPs throttle.
2.  **Analyze** Heat Sources: Sensor (Analog), SerDes (High Speed), ISP (Compute).
3.  **Implement** Thermal Throttling Strategies: Reduce FPS, Reduce Resolution, Disable Algorithms.
4.  **Configure** the Linux Thermal Framework: Thermal Zones, Cooling Devices, Governors.
5.  **Design** for Reliability: MTBF (Mean Time Between Failures), De-rating.
6.  **Perform** a Thermal Stress Test.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera System, IR Thermometer or Thermal Camera.
*   **Software:** `thermal-engine` (Qualcomm) or Standard Linux Thermal Sysfs.
*   **Knowledge:** Heat Transfer (Conduction, Convection), Junction Temperature ($T_j$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Heat & Image Quality
*   **Dark Current:** Doubles for every ~7°C increase in temperature.
*   **Hot Pixels:** Defective pixels become visible as bright spots at high temps.
*   **Lens Shift:** Thermal expansion changes the focal length -> Defocus (Thermal Drift).
*   **ISP Limit:** If the SoC gets too hot ($> 85^\circ C$), it may crash or damage silicon.

### 🔹 Part 2: Linux Thermal Framework
*   **Thermal Zone:** A sensor measuring temperature (e.g., `cpu-thermal`, `camera-thermal`).
*   **Cooling Device:** Something that can reduce heat (e.g., `fan`, `cpufreq`, `camera-fps`).
*   **Trip Points:**
    *   **Passive:** Start throttling (soft limit).
    *   **Critical:** Shut down immediately (hard limit).
*   **Governor:** Logic that decides how much cooling to apply (Step-Wise, PID).

### 🔹 Part 3: Reliability Metrics
*   **HTOL (High Temperature Operating Life):** Running at 125°C for 1000 hours.
*   **Temperature Cycling:** -40°C to +105°C repeatedly.
*   **De-rating:** Using components below their max rating (e.g., using a 10V capacitor on a 5V line) to increase lifespan.

---

## 💻 Implementation Examples

### Example 1: Defining a Thermal Zone (Device Tree)

Mapping a temperature sensor to a cooling map.

```dts
thermal-zones {
    camera_thermal: camera-thermal {
        polling-delay-passive = <1000>; /* Check every 1s */
        polling-delay = <5000>;
        
        thermal-sensors = <&tsens 5>; /* Sensor ID */
        
        trips {
            camera_warm: trip-point-0 {
                temperature = <65000>; /* 65 C */
                hysteresis = <2000>;
                type = "passive";
            };
            camera_hot: trip-point-1 {
                temperature = <85000>; /* 85 C */
                hysteresis = <2000>;
                type = "critical";
            };
        };
        
        cooling-maps {
            map0 {
                trip = <&camera_warm>;
                cooling-device = <&camera_dev 1 2>; /* Level 1 to 2 */
            };
        };
    };
};
```

### Example 2: Implementing a Cooling Device (Driver)

The Camera Driver registers as a cooling device.

```c
/* Set Cooling State Callback */
static int cam_set_cur_state(struct thermal_cooling_device *cdev, unsigned long state)
{
    struct cam_device *cam = cdev->devdata;
    
    switch (state) {
        case 0: // Normal
            cam->fps_limit = 30;
            break;
        case 1: // Warm - Throttle
            cam->fps_limit = 15;
            dev_warn(cam->dev, "Thermal Throttling: 15 FPS\n");
            break;
        case 2: // Hot - Stop
            cam->fps_limit = 0; // Stop Streaming
            dev_err(cam->dev, "Thermal Shutdown\n");
            break;
    }
    return 0;
}

static const struct thermal_cooling_device_ops cam_cooling_ops = {
    .get_max_state = cam_get_max_state,
    .get_cur_state = cam_get_cur_state,
    .set_cur_state = cam_set_cur_state,
};

/* In Probe */
cam->cdev = thermal_cooling_device_register("camera-cooling", cam, &cam_cooling_ops);
```

### Example 3: Userspace Thermal Monitor

Script to log temps.

```bash
#!/bin/bash
while true; do
    TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
    echo "$(date) - CPU Temp: $((TEMP/1000)) C"
    
    # Check Camera Temp (if available via I2C)
    # i2cget ...
    
    sleep 1
done
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Inducing Thermal Throttling

**Objective:** Verify the safety mechanism.

**Steps:**
1.  Run the Camera at 4K @ 60fps (High Load).
2.  Use a heat gun (carefully!) or block airflow to heat up the SoC.
3.  **Monitor:** `dmesg` for throttling warnings.
4.  **Observation:** FPS should drop to 30, then 15, as temp rises.

### Lab 2: Dark Current Calibration

**Objective:** See the noise.

**Steps:**
1.  Cover the lens (Black frame).
2.  Capture an image at Room Temp (25°C).
3.  Heat the sensor to 60°C.
4.  Capture an image.
5.  **Compare:** The hot image will be gray/purple instead of black due to thermal noise.
6.  **Fix:** Enable "Black Level Correction" (BLC) in the ISP, which subtracts the measured black level dynamically.

### Lab 3: Power vs Temp

**Objective:** Correlation.

**Steps:**
1.  Measure Power (Day 71) and Temp simultaneously.
2.  **Observation:** As silicon gets hotter, leakage current increases, which generates *more* heat. This is "Thermal Runaway".

---

## 🐛 Debugging Thermal Issues

### Debug 1: "Purple Haze" in Corners

**Symptom:** Corners of the image turn purple.

**Cause:**
*   **Lens Shading:** The lens mount expanded, changing the shading profile.
*   **Sensor Heat:** The corners of the sensor might be hotter (near voltage regulators).
*   **Fix:** Dynamic Lens Shading Correction (LSC) that adapts to temperature.

### Debug 2: Focus Drift

**Symptom:** Image becomes blurry after 10 mins.

**Cause:**
*   Plastic lens barrel expanding.
*   **Fix:** Use Glass lenses (Athermal) or an Actuator (VCM) with thermal compensation logic.

---

## ⚡ Performance Optimization

### Optimization 1: Computational Offload

*   If the ISP is overheating, offload some tasks (e.g., JPEG encoding) to the CPU or DSP if they are cooler.
*   Spread the heat across the die.

### Optimization 2: Intelligent Throttling

*   Instead of dropping FPS (which looks bad), drop **Resolution** or **Bitrate**.
*   Or disable heavy algorithms (e.g., disable "Beauty Mode" or "Noise Reduction" temporarily).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is $T_j$ (Junction Temp) vs $T_a$ (Ambient Temp)?** ($T_j$ is the silicon temp, $T_a$ is the air temp. $T_j = T_a + Power \times \theta_{JA}$).
2.  **Why does "Dark Current" matter for low-light performance?** (It sets the noise floor. You can't see signals weaker than the dark current).
3.  **What is a "Cooling Device" in Linux?** (An abstraction for anything that can reduce temp).
4.  **Why do automotive cameras use metal housings?** (Heatsink functionality).

### Practical Challenges

1.  **Implement "Temp-Based BLC":** Read the sensor's internal temp sensor via I2C. Adjust the Black Level subtraction value linearly based on temp.
2.  **Design a Thermal Test Plan:** Define the test duration, temp range, and pass/fail criteria for a new camera product.

---

## 📚 Further Reading & Resources

### Documentation
*   **Linux Thermal Subsystem Documentation.**
*   **JEDEC Standards (JESD51) for Thermal Measurement.**

---

## 🎓 Summary

Today we covered:
- ✅ **Heat:** The enemy of image quality.
- ✅ **Framework:** Thermal Zones & Cooling.
- ✅ **Throttling:** Graceful degradation.
- ✅ **Reliability:** Designing for the extreme.
- ✅ **Drift:** Focus and Color shifts.

**Next:** Day 74 - Phase 3 Capstone Project: Design & Architecture.

---

**Day 73 Complete** | Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project
