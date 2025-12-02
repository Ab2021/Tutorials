# Day 163: Power Management (Regulators & PMIC)
## Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power

---

## 🎯 Learning Objectives
1.  **Understand** the Power Tree of a Camera System (SoC, Sensor, SerDes).
2.  **Control** Power Rails using Linux Regulators (`regulator-fixed`, `regulator-gpio`).
3.  **Implement** Runtime Power Management (Runtime PM) to suspend idle devices.
4.  **Measure** Power Consumption using `ina219` or `powertop`.
5.  **Optimize** Clock Gating and Voltage Scaling (DVFS).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Jetson/Pi, Multimeter, INA219 Power Sensor.
*   **Software:** Linux Kernel Source, Device Tree.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Power Tree
*   **PMIC (Power Management IC):** Converts Battery Voltage (12V) to System Voltages.
*   **Rails:**
    *   **Core:** 0.8V (SoC Logic). High Current.
    *   **IO:** 1.8V / 3.3V (GPIO, I2C).
    *   **Analog (AVDD):** 2.8V (Image Sensor Pixel Array). Clean noise-free power.
    *   **Digital (DVDD):** 1.2V (Image Sensor Logic).
    *   **Interface (DOVDD):** 1.8V (MIPI PHY).

### 🔹 Part 2: Linux Regulator Framework
*   **Consumers:** Drivers (e.g., `imx219.c`) request power (`regulator_enable`).
*   **Providers:** PMIC Drivers control the hardware.
*   **Constraints:** Defined in Device Tree (Min/Max Voltage).

### 🔹 Part 3: Runtime PM
*   **Concept:** If a device is not used for N seconds, turn it off.
*   **Autosuspend:** The driver saves context, disables clocks, disables regulators.
*   **Resume:** When `open()` is called, restore context.

---

## 💻 Implementation Examples

### Example 1: Device Tree Regulator Definition

Defining a fixed 2.8V regulator controlled by a GPIO.

```dts
/ {
    cam_avdd_2v8: regulator-cam-avdd {
        compatible = "regulator-fixed";
        regulator-name = "cam_avdd_2v8";
        regulator-min-microvolt = <2800000>;
        regulator-max-microvolt = <2800000>;
        gpio = <&gpio TEGRA_GPIO(S, 4) GPIO_ACTIVE_HIGH>;
        enable-active-high;
    };
};

&i2c0 {
    imx219: imx219@10 {
        compatible = "sony,imx219";
        // ...
        vana-supply = <&cam_avdd_2v8>; // Link to regulator
    };
};
```

### Example 2: Driver Implementation (Runtime PM)

```c
#include <linux/pm_runtime.h>

static int imx219_probe(struct i2c_client *client)
{
    // ...
    pm_runtime_set_active(&client->dev);
    pm_runtime_enable(&client->dev);
    // ...
}

static int imx219_runtime_suspend(struct device *dev)
{
    struct i2c_client *client = to_i2c_client(dev);
    struct imx219 *sensor = i2c_get_clientdata(client);

    regulator_disable(sensor->vana);
    clk_disable_unprepare(sensor->xclk);
    
    return 0;
}

static int imx219_runtime_resume(struct device *dev)
{
    // Enable Clocks and Regulators
    // Restore Registers
    return 0;
}

static const struct dev_pm_ops imx219_pm_ops = {
    SET_RUNTIME_PM_OPS(imx219_runtime_suspend, imx219_runtime_resume, NULL)
};
```

### Example 3: Measuring Power (Python + INA219)

```python
from ina219 import INA219
from ina219 import DeviceRangeError

SHUNT_OHMS = 0.1
ina = INA219(SHUNT_OHMS)
ina.configure()

try:
    print("Bus Voltage: %.3f V" % ina.voltage())
    print("Current: %.3f mA" % ina.current())
    print("Power: %.3f mW" % ina.power())
except DeviceRangeError as e:
    print(e)
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Power Profiling with `powertop`

**Objective:** Find the battery drainers.

**Steps:**
1.  Run `sudo powertop --calibrate`.
2.  Observe the "Tunables" tab.
3.  **Action:** Enable "Autosuspend" for USB devices and PCI.
4.  **Result:** Power consumption drops by 1-2 Watts.

### Lab 2: Control the Regulator

**Objective:** Manual control.

**Steps:**
1.  Find the regulator in sysfs: `/sys/class/regulator/`.
2.  Check status: `cat state`.
3.  **Experiment:** Try to disable it while the camera is streaming.
4.  **Result:** The kernel should prevent you (Reference Counting). "Device or resource busy".

### Lab 3: DVFS (Dynamic Voltage and Frequency Scaling)

**Objective:** Slow down to save power.

**Steps:**
1.  Check current frequency: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq`.
2.  Set Governor to "Powersave": `echo powersave > scaling_governor`.
3.  **Measure:** Power drops. Latency increases.
4.  **Task:** Find the lowest frequency that still sustains 30fps.

---

## 🐛 Debugging Power Issues

### Debug 1: "Camera Not Found" after Resume

**Symptom:** Camera works once. After suspend/resume, it vanishes.

**Cause:**
*   **Register Loss:** The sensor lost its configuration when power was cut.
*   **Fix:** The `runtime_resume` function MUST re-initialize all registers (re-send the init sequence).

### Debug 2: "Brownout"

**Symptom:** System reboots when turning on the camera.

**Cause:**
*   **Inrush Current:** Turning on the regulator causes a spike.
*   **Fix:** Use Soft Start (if PMIC supports it) or add bulk capacitors near the sensor.

---

## ⚡ Performance Optimization

### Optimization 1: Clock Gating

*   Disable the ISP clock when no frame is being processed.
*   Modern SoCs do this automatically, but ensure your driver calls `clk_prepare_enable` only when streaming.

### Optimization 2: Low Power Modes (LP1/LP2)

*   Jetson supports Deep Sleep (SC7).
*   Ensure your camera driver supports `suspend` and `resume` callbacks to allow the system to enter SC7.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is a "Buck Converter" vs "LDO"?** (Buck = Efficient switching regulator. LDO = Linear regulator, clean but inefficient (burns heat)).
2.  **Why do we need 3 different voltages for a sensor?** (Analog needs clean power. Digital needs low voltage for speed/efficiency. IO needs to match the host voltage).
3.  **What is "Reference Counting" in regulators?** (If 3 drivers share a regulator, it stays ON until all 3 disable it).

### Practical Challenges

1.  **Script a Power Logger:** Write a script that logs power every 100ms to a CSV file while running a heavy workload. Plot the graph.
2.  **Implement "Wake-on-Motion":** Configure the IMX sensor to generate an interrupt on motion (if supported) to wake the SoC.

---

## 📚 Further Reading & Resources

### Documentation
*   **Linux Regulator Framework.**
*   **Jetson Power Management Guide.**

---

## 🎓 Summary

Today we covered:
- ✅ **Power Tree:** AVDD, DVDD, DOVDD.
- ✅ **Regulators:** Kernel control.
- ✅ **Runtime PM:** Suspend/Resume.
- ✅ **Measurement:** INA219.
- ✅ **Efficiency:** DVFS and Clock Gating.

**Next:** Day 164 - Thermal Management.

---

**Day 163 Complete** | Phase 3: Camera Systems & ISP | Week 26: Performance Optimization & Power


