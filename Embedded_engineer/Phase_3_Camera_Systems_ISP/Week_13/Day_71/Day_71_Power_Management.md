# Day 71: Power Management (Suspend/Resume)
## Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project

---

## 🎯 Learning Objectives
1.  **Understand** Camera Power Rails: AVDD (Analog), DOVDD (IO), DVDD (Digital).
2.  **Analyze** Power Sequencing: The strict order of turning on/off voltages.
3.  **Implement** Runtime Power Management (Runtime PM) in Linux Drivers.
4.  **Handle** System Suspend (S3) and Resume in the Camera Driver.
5.  **Optimize** Sensor Low Power Modes (Standby vs Power Down).
6.  **Measure** Power Consumption using a Power Monitor (e.g., Monsoon or INA219).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera Module, Oscilloscope (to check sequencing), Power Monitor.
*   **Software:** Linux Kernel Source, `powertop`.
*   **Knowledge:** Linux Device Driver Model (`probe`, `remove`, `suspend`, `resume`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Three Rails
Camera sensors typically require 3 separate power supplies:
1.  **AVDD (2.8V):** Analog power for the Pixel Array. Cleanest supply required (High PSRR LDO). Noise here = Image Noise.
2.  **DOVDD (1.8V):** I/O power for MIPI/I2C. Matches the SoC logic level.
3.  **DVDD (1.2V):** Digital power for the ISP/Logic core. Often generated internally by the sensor or supplied externally for efficiency.

### 🔹 Part 2: Power Sequencing
*   Sensors are sensitive. Turning on signals (I2C/MIPI) before power can latch-up the chip.
*   **Typical Sequence (Power Up):**
    1.  Turn on DOVDD -> Wait.
    2.  Turn on AVDD -> Wait.
    3.  Turn on DVDD -> Wait.
    4.  Enable MCLK (Master Clock).
    5.  Assert RESET_N (High).
*   **Power Down:** Reverse order.

### 🔹 Part 3: Linux Power Management
*   **Runtime PM:** Turns off the device *while the system is running* if it's idle (e.g., Camera app closed).
*   **System Suspend (S3):** The whole system goes to sleep (RAM retention). Camera must power down completely and restore state on resume.

---

## 💻 Implementation Examples

### Example 1: Defining Regulators in Device Tree

```dts
camera_sensor: camera@1a {
    compatible = "sony,imx219";
    reg = <0x1a>;
    
    /* Power Supplies */
    avdd-supply = <&vreg_l10a_2p8>;
    dovdd-supply = <&vreg_l5a_1p8>;
    dvdd-supply = <&vreg_l2a_1p2>;
    
    /* GPIOs */
    reset-gpios = <&tlmm 10 GPIO_ACTIVE_LOW>;
    clocks = <&cam_clk>;
};
```

### Example 2: Power Sequence in Driver

```c
static int imx219_power_on(struct device *dev)
{
    struct imx219 *sensor = to_imx219(dev);
    int ret;

    /* 1. Enable Regulators (Order matters!) */
    ret = regulator_enable(sensor->dovdd); // 1.8V
    usleep_range(1000, 2000);
    
    ret = regulator_enable(sensor->avdd); // 2.8V
    usleep_range(1000, 2000);
    
    ret = regulator_enable(sensor->dvdd); // 1.2V
    usleep_range(5000, 6000);

    /* 2. Enable Clock */
    clk_prepare_enable(sensor->xclk);
    usleep_range(5000, 6000);

    /* 3. Release Reset */
    gpiod_set_value_cansleep(sensor->reset_gpio, 0); // Active Low -> 0 means High (Release)
    usleep_range(5000, 6000);

    return 0;
}

static int imx219_power_off(struct device *dev)
{
    struct imx219 *sensor = to_imx219(dev);
    
    /* Reverse Order */
    gpiod_set_value_cansleep(sensor->reset_gpio, 1); // Assert Reset
    clk_disable_unprepare(sensor->xclk);
    regulator_disable(sensor->dvdd);
    regulator_disable(sensor->avdd);
    regulator_disable(sensor->dovdd);
    
    return 0;
}
```

### Example 3: Runtime PM Implementation

```c
/* Called when app opens camera */
static int imx219_s_stream(struct v4l2_subdev *sd, int enable)
{
    struct i2c_client *client = v4l2_get_subdevdata(sd);
    
    if (enable) {
        pm_runtime_get_sync(&client->dev); // Wakes up device
        // Write registers to start streaming
    } else {
        // Write registers to stop streaming
        pm_runtime_put(&client->dev); // Allows device to sleep
    }
    return 0;
}

/* PM Callbacks */
static int imx219_runtime_suspend(struct device *dev)
{
    return imx219_power_off(dev);
}

static int imx219_runtime_resume(struct device *dev)
{
    return imx219_power_on(dev);
}

static const struct dev_pm_ops imx219_pm_ops = {
    SET_RUNTIME_PM_OPS(imx219_runtime_suspend, imx219_runtime_resume, NULL)
    SET_SYSTEM_SLEEP_PM_OPS(imx219_suspend, imx219_resume)
};
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Measuring Power Consumption

**Objective:** Quantify the cost of the camera.

**Steps:**
1.  Connect an external power monitor to the Camera Module power pins.
2.  Measure Current in **Power Down** (Should be < 10uA).
3.  Measure Current in **Streaming** (e.g., 200mA).
4.  **Calculate:** Power = Voltage x Current.
5.  **Observation:** High resolution/framerate increases digital power (DVDD) significantly.

### Lab 2: Verifying Sequence with Scope

**Objective:** Prevent Latch-up.

**Steps:**
1.  Probe DOVDD (Ch1) and AVDD (Ch2).
2.  Trigger on DOVDD rising edge.
3.  **Goal:** DOVDD must rise *before* AVDD.
4.  **Fail:** If AVDD rises first, the ESD diodes on the IO lines might conduct, damaging the sensor.

### Lab 3: Testing Runtime PM

**Objective:** Save battery.

**Steps:**
1.  Boot system. Camera not used.
2.  Check `regulator_summary` in debugfs. Camera regulators should be OFF (0 microvolts).
3.  Start `v4l2-ctl --stream-mmap`.
4.  Check regulators. Should be ON.
5.  Stop stream.
6.  Check regulators. Should turn OFF after autosuspend delay (usually 1-2s).

---

## 🐛 Debugging Power Issues

### Debug 1: "I2C Timeout" on Boot

**Symptom:** Driver fails to probe.

**Cause:**
*   Power sequence violation. Sensor is in latch-up or reset state.
*   MCLK not enabled before I2C access.
*   **Fix:** Check the datasheet for "Power-On Sequence" timing diagrams. Add `usleep` delays.

### Debug 2: High Leakage Current

**Symptom:** Battery drains even when camera is off.

**Cause:**
*   Floating input pins. If the sensor is powered down but the SoC drives the I2C/GPIO lines High, current leaks through the protection diodes.
*   **Fix:** Configure SoC pins as "Input" or "Low" during suspend (`pinctrl-sleep`).

---

## ⚡ Performance Optimization

### Optimization 1: Dynamic Voltage Scaling (DVS)

*   If running at low framerate (binning), the sensor might operate at lower DVDD (e.g., 1.0V instead of 1.2V).
*   Driver can adjust the regulator voltage based on the mode.

### Optimization 2: Clock Gating

*   Stop the MCLK (Ext Clock) immediately when entering Low Power mode.
*   Saves SoC power.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do we need 3 different voltages?** (Physics of transistors: Analog needs clean high voltage, Digital needs low voltage for speed/power).
2.  **What is "Dark Current"?** (Noise generated by heat/leakage in the pixel).
3.  **Why must I/O voltage (DOVDD) match the SoC?** (To avoid logic level mismatch).
4.  **What happens if you pull RESET_N low?** (Sensor enters Reset state, usually lowest power).

### Practical Challenges

1.  **Implement "Deep Sleep":** Some sensors have a register-based Standby mode that keeps settings but stops streaming. Implement this in `s_stream(0)` instead of full power off to wake up faster.
2.  **Debug a "Hot" Camera:** If the module feels hot to the touch (> 60°C), investigate if LDOs are oscillating or if there is a short.

---

## 📚 Further Reading & Resources

### Documentation
*   **Linux Power Management Guide.**
*   **Sensor Datasheet (Power Consumption section).**

---

## 🎓 Summary

Today we covered:
- ✅ **Rails:** AVDD, DOVDD, DVDD.
- ✅ **Sequencing:** The golden rule of power up.
- ✅ **Runtime PM:** Auto-sleep.
- ✅ **Measurement:** Verifying consumption.
- ✅ **Leakage:** Pin states during sleep.

**Next:** Day 72 - Fast Boot & Latency Optimization.

---

**Day 71 Complete** | Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project
