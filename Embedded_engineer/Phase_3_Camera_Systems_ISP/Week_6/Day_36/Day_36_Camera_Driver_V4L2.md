# Day 36: Camera Driver Development (V4L2 Deep Dive)
## Phase 3: Camera Systems & ISP | Week 6: Image Quality & Tuning

---

## 🎯 Learning Objectives
1.  **Master** the Linux V4L2 Sub-device (Subdev) framework.
2.  **Implement** a Sensor Driver (I2C) with `v4l2_subdev_ops`.
3.  **Configure** the Media Controller Topology (Sensor -> CSI -> ISP).
4.  **Handle** Controls (Gain, Exposure, H/V Flip) via `v4l2_ctrl_handler`.
5.  **Debug** driver probing, linking, and streaming issues.
6.  **Understand** Device Tree (DTS) bindings for cameras.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Linux Development Board (Raspberry Pi, Jetson), Image Sensor.
*   **Software:** Linux Kernel Source, `v4l2-utils` (`media-ctl`, `v4l2-ctl`).
*   **Knowledge:** Linux Kernel Modules, I2C Subsystem.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The V4L2 Architecture
*   **Video Device (`/dev/videoX`):** The user-space interface for streaming buffers.
*   **Sub-Device (`/dev/v4l-subdevX`):** Represents internal components (Sensor, CSI-RX, ISP).
*   **Media Controller (`/dev/mediaX`):** Manages the links between sub-devices.

### 🔹 Part 2: The Sensor Driver
A camera sensor is an I2C slave that outputs data via CSI-2.
*   **Probe:** Detect chip ID, enable regulators/clocks.
*   **s_stream:** Start/Stop streaming (write registers).
*   **get_fmt / set_fmt:** Negotiate resolution and format (Bayer, YUV).
*   **Controls:** Expose Gain/Exposure to user-space.

### 🔹 Part 3: Device Tree (DTS)
Describes the hardware connection.
*   **Port/Endpoint:** Describes the CSI-2 lanes and clock.
*   **Clocks:** The `mclk` (Master Clock) provided by the SoC.
*   **GPIOs:** Reset (`reset-gpios`) and Power Down (`pwdn-gpios`).

---

## 💻 Implementation Examples

### Example 1: Basic Sensor Driver Skeleton

```c
/**
 * @file my_sensor.c
 * @brief Simple V4L2 Subdev Driver
 */

#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/v4l2-subdev.h>
#include <media/v4l2-ctrls.h>

struct my_sensor {
    struct i2c_client *client;
    struct v4l2_subdev sd;
    struct v4l2_ctrl_handler ctrls;
    struct mutex lock;
    bool streaming;
};

static int my_sensor_s_stream(struct v4l2_subdev *sd, int enable)
{
    struct my_sensor *sensor = container_of(sd, struct my_sensor, sd);
    struct i2c_client *client = sensor->client;
    
    if (enable) {
        // Write Registers to Start Streaming
        // i2c_write(client, REG_MODE, MODE_STREAM);
    } else {
        // Stop Streaming
        // i2c_write(client, REG_MODE, MODE_STANDBY);
    }
    sensor->streaming = enable;
    return 0;
}

static const struct v4l2_subdev_video_ops my_sensor_video_ops = {
    .s_stream = my_sensor_s_stream,
};

static const struct v4l2_subdev_ops my_sensor_subdev_ops = {
    .video = &my_sensor_video_ops,
};

static int my_sensor_probe(struct i2c_client *client)
{
    struct my_sensor *sensor;
    
    sensor = devm_kzalloc(&client->dev, sizeof(*sensor), GFP_KERNEL);
    v4l2_i2c_subdev_init(&sensor->sd, client, &my_sensor_subdev_ops);
    
    // Initialize Controls (Gain, Exposure)
    v4l2_ctrl_handler_init(&sensor->ctrls, 2);
    v4l2_ctrl_new_std(&sensor->ctrls, NULL, V4L2_CID_GAIN, 0, 255, 1, 0);
    v4l2_ctrl_new_std(&sensor->ctrls, NULL, V4L2_CID_EXPOSURE, 0, 65535, 1, 100);
    sensor->sd.ctrl_handler = &sensor->ctrls;
    
    return v4l2_async_register_subdev(&sensor->sd);
}

static const struct i2c_device_id my_sensor_id[] = {
    { "my_sensor", 0 },
    { }
};
MODULE_DEVICE_TABLE(i2c, my_sensor_id);

static struct i2c_driver my_sensor_driver = {
    .driver = { .name = "my_sensor" },
    .probe = my_sensor_probe,
    .id_table = my_sensor_id,
};
module_i2c_driver(my_sensor_driver);
MODULE_LICENSE("GPL");
```

### Example 2: Device Tree Snippet

```dts
/* In the SoC dtsi or board dts */

&i2c0 {
    my_sensor: camera@1a {
        compatible = "vendor,my-sensor";
        reg = <0x1a>;
        clocks = <&clk_cam>;
        clock-names = "mclk";
        reset-gpios = <&gpio 10 GPIO_ACTIVE_LOW>;
        
        port {
            my_sensor_out: endpoint {
                remote-endpoint = <&csi_in>;
                data-lanes = <1 2>;
                clock-lanes = <0>;
            };
        };
    };
};

&csi {
    port {
        csi_in: endpoint {
            remote-endpoint = <&my_sensor_out>;
            data-lanes = <1 2>;
        };
    };
};
```

### Example 3: Media Controller Configuration (User Space)

Using `media-ctl` to link the pipeline.

```bash
# 1. Reset
media-ctl -r

# 2. Configure Sensor Output (1920x1080 Raw10)
media-ctl -V "'my_sensor 0-001a':0 [fmt:SRGGB10_1X10/1920x1080 field:none]"

# 3. Configure CSI Receiver (VI)
media-ctl -V "'tegra-vi':0 [fmt:SRGGB10_1X10/1920x1080 field:none]"

# 4. Enable Link
media-ctl -l "'my_sensor 0-001a':0 -> 'tegra-vi':0 [1]"
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Probing the Driver

**Objective:** Verify the driver loads.

**Steps:**
1.  Compile the kernel module (`.ko`).
2.  `insmod my_sensor.ko`.
3.  Check `dmesg`. Look for "my_sensor: probe success".
4.  Check `ls /dev/v4l-subdev*`.
5.  **Failure:** If probe fails, check I2C address (0x1a vs 0x34?), Power (Regulators), and Reset GPIO.

### Lab 2: Streaming Data

**Objective:** Capture a raw frame.

**Steps:**
1.  Configure pipeline (Example 3).
2.  Start streaming with `v4l2-ctl`:
    ```bash
    v4l2-ctl --set-fmt-video=width=1920,height=1080,pixelformat=RG10 --stream-mmap --stream-count=1 --stream-to=frame.raw
    ```
3.  **Debug:** If it hangs, check interrupts (`cat /proc/interrupts`). If count is 0, the CSI hardware is not receiving data (MIPI lanes issue).

### Lab 3: Changing Controls

**Objective:** Verify Gain/Exposure.

**Steps:**
1.  Start streaming (background).
2.  Change Gain:
    ```bash
    v4l2-ctl --set-ctrl gain=200
    ```
3.  **Observation:** The image should get brighter immediately.
4.  **Code:** Verify that `.s_ctrl` in your driver is called and writes to the I2C register.

---

## 🐛 Debugging Techniques

### Debug 1: "Broken Pipe" or "Input/Output Error"

**Symptom:** `v4l2-ctl` fails immediately.

**Cause:**
*   Pipeline not linked (Media Controller).
*   Format mismatch (Sensor outputs RGGB, CSI expects BGGR).
*   **Fix:** Check `media-ctl -p` topology.

### Debug 2: Frame Timeout (Select Timeout)

**Symptom:** Application waits for frame but it never comes.

**Cause:**
*   Sensor is not outputting data (Clock missing, Reset held).
*   MIPI Lane configuration wrong (2 lanes vs 4 lanes).
*   **Fix:** Probe the MIPI lines with an oscilloscope (Differential probe). You should see high-speed activity.

---

## ⚡ Performance Optimization

### Optimization 1: Async Probing

*   Sensors often depend on PMICs or Clocks that might load *after* the camera driver.
*   Use `v4l2_async_register_subdev`. The V4L2 core will wait until all dependencies (endpoints) are ready before creating the `/dev/videoX` node.

### Optimization 2: Regmap for I2C

*   Don't use raw `i2c_transfer`. Use `regmap_i2c`.
*   Provides caching (read-back without I2C traffic) and debugfs support (`/sys/kernel/debug/regmap`).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the role of the "Media Controller" in V4L2?**
2.  **Why do we need a "Subdev" for the sensor? Why not just one video device?**
3.  **What is the difference between `s_stream` and `s_power`?**
4.  **How does the kernel know which I2C bus the camera is on?**

### Practical Challenges

1.  **Implement a "Test Pattern" control:** Add a V4L2 control that switches the sensor's internal test pattern generator (Color Bars).
2.  **Add "Frame Rate" control:** Implement `frame_interval` ops to allow changing FPS (adjusting V-Blanking).

---

## 📚 Further Reading & Resources

### Documentation
*   **Linux Kernel:** `Documentation/driver-api/media/v4l2-subdev.rst`.
*   **V4L2 API Spec:** The official userspace API guide.

### Code
*   **Drivers:** `drivers/media/i2c/imx219.c` (Excellent reference).

---

## 🎓 Summary

Today we covered:
- ✅ **V4L2 Architecture:** Video, Subdev, Media.
- ✅ **Driver Implementation:** Probe, Stream, Controls.
- ✅ **Device Tree:** Describing the hardware graph.
- ✅ **Media Controller:** Linking the pipeline.
- ✅ **Debugging:** Solving "No Video" issues.

**Next:** Day 37 - Android Camera HAL (HAL3) Overview.

---

**Day 36 Complete** | Phase 3: Camera Systems & ISP | Week 6: Image Quality & Tuning
