# Day 64: EVS Camera Integration
## Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS

---

## 🎯 Learning Objectives
1.  **Implement** Camera Metadata (Static and Dynamic) in EVS.
2.  **Handle** Camera Parameters (Brightness, Contrast, Auto-Exposure).
3.  **Support** multiple camera streams (e.g., Rear + Trailer).
4.  **Integrate** with the Vehicle HAL (VHAL) to trigger camera events (Reverse Gear).
5.  **Manage** Camera Hot-plugging (USB Cameras).
6.  **Debug** Metadata synchronization issues.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with controllable ISP (or UVC camera).
*   **Software:** EVS HAL source code.
*   **Knowledge:** Android Camera Metadata tags (`ANDROID_SENSOR_EXPOSURE_TIME`, etc.).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Metadata in EVS
*   Unlike standard Android Camera HAL (which uses a complex `CameraMetadata` object), EVS uses a simpler metadata structure, but can optionally support the full Android metadata tags.
*   **Static Metadata:** Capabilities (Resolution, Focal Length, Lens Position). Sent once during `getCameraList`.
*   **Dynamic Metadata:** Per-frame status (Exposure time used, Gain used, AE State). Sent with each frame in `BufferDesc`.

### 🔹 Part 2: Camera Parameters
*   EVS defines a set of standard parameters (`CameraParam` enum):
    *   `BRIGHTNESS`, `CONTRAST`, `AUTOGAIN`, `AF_MODE`.
*   The HAL translates these set requests into V4L2 controls (`VIDIOC_S_CTRL`).

### 🔹 Part 3: VHAL Integration
*   The EVS HAL usually doesn't listen to CAN bus directly.
*   The **EVS Manager** or **EVS App** listens to VHAL properties (`GEAR_SELECTION`).
*   When Reverse is detected, the App calls `openCamera`.
*   **Turn Signal:** Left Turn signal -> Open Left Side Camera.

---

## 💻 Implementation Examples

### Example 1: Handling Parameters

Mapping EVS parameters to V4L2 controls.

```cpp
Return<void> EvsCamera::setIntParameter(CameraParam id, int32_t value) {
    struct v4l2_control ctrl = {};
    
    switch (id) {
        case CameraParam::BRIGHTNESS:
            ctrl.id = V4L2_CID_BRIGHTNESS;
            break;
        case CameraParam::CONTRAST:
            ctrl.id = V4L2_CID_CONTRAST;
            break;
        case CameraParam::AUTO_EXPOSURE:
            ctrl.id = V4L2_CID_EXPOSURE_AUTO;
            // Map EVS value (0/1) to V4L2 enum
            value = (value == 1) ? V4L2_EXPOSURE_AUTO : V4L2_EXPOSURE_MANUAL;
            break;
        default:
            return Void(); // Not supported
    }
    
    ctrl.value = value;
    if (ioctl(mVideoFd, VIDIOC_S_CTRL, &ctrl) < 0) {
        ALOGE("Failed to set control %d", ctrl.id);
    }
    return Void();
}
```

### Example 2: Adding Metadata to Frame

Attaching exposure info to the buffer.

```cpp
// In capture thread
BufferDesc evsBuf;
// ... fill bufferId ...

// Resize metadata vector
evsBuf.metadata.resize(2); 

// 1. Exposure Time
evsBuf.metadata[0] = {
    .tag = ANDROID_SENSOR_EXPOSURE_TIME,
    .value = { .int64Values = { current_exposure_ns } }
};

// 2. Sensitivity (ISO)
evsBuf.metadata[1] = {
    .tag = ANDROID_SENSOR_SENSITIVITY,
    .value = { .int32Values = { current_iso } }
};

mReceiver->deliverFrame(evsBuf);
```

### Example 3: Hotplug Detection (UVC)

Using `inotify` to detect `/dev/videoX` creation.

```cpp
void EvsEnumerator::monitorDevices() {
    int fd = inotify_init();
    inotify_add_watch(fd, "/dev/", IN_CREATE | IN_DELETE);
    
    while (true) {
        read(fd, buffer, SIZE);
        // If /dev/videoX created:
        // 1. Check if it's a camera (ioctl QUERYCAP).
        // 2. Add to internal list.
        // 3. Notify EVS Manager (via callback if supported, or just update list).
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Controlling Brightness

**Objective:** Verify parameter control.

**Steps:**
1.  Start EVS App.
2.  Use `service call` to invoke `setIntParameter` manually (or modify the App to add a slider).
    ```bash
    # Example (Pseudo-command, depends on interface hash)
    adb shell service call android.hardware.automotive.evs.IEvsCamera/default ...
    ```
3.  **Observation:** Image should get brighter/darker.

### Lab 2: Reverse Gear Logic

**Objective:** Automate camera start.

**Steps:**
1.  Modify `evs_app` to subscribe to VHAL.
2.  In `onPropertyChange`:
    ```cpp
    if (propId == GEAR_SELECTION && value == REVERSE) {
        startCamera();
    } else if (value == DRIVE) {
        stopCamera();
    }
    ```
3.  Inject event via `adb`.
4.  **Result:** Camera starts automatically.

### Lab 3: Metadata Logging

**Objective:** Verify frame metadata.

**Steps:**
1.  In `evs_app`, print the metadata of incoming frames.
2.  Cover the lens. Auto-Exposure should increase exposure time.
3.  **Observation:** The printed `EXPOSURE_TIME` value should increase.

---

## 🐛 Debugging Techniques

### Debug 1: "Control Failed"

**Symptom:** `setIntParameter` logs errors.

**Cause:**
*   The V4L2 driver does not support that specific control (e.g., Manual Exposure on a cheap webcam).
*   **Fix:** Run `v4l2-ctl -d /dev/video0 -L` to list supported controls. Update HAL to match.

### Debug 2: Metadata Sync

**Symptom:** Metadata lags behind video (e.g., ISO updates 3 frames late).

**Cause:**
*   ISP pipeline depth. The settings applied now take effect 3 frames later.
*   **Fix:** The Driver must queue the settings and match them to the correct frame sequence number (Request ID).

---

## ⚡ Performance Optimization

### Optimization 1: Cached Parameters

*   Don't call `ioctl(VIDIOC_S_CTRL)` if the value hasn't changed.
*   Cache the current value in the HAL.
*   `ioctl` is a system call (Context Switch) and is expensive.

### Optimization 2: Batch Metadata

*   Instead of allocating a new `std::vector` for metadata every frame, reuse a pre-allocated vector.
*   Memory allocation in the hot loop causes jitter.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is "Dynamic Metadata" important for Image Quality?** (Allows the ISP to report what it *actually* did vs what was requested).
2.  **How does EVS handle multiple cameras (Surround View)?** (Opens 4 `IEvsCamera` instances).
3.  **What happens if the VHAL crashes?** (EVS App should detect binder death and restart/wait).
4.  **Why use `inotify` for hotplug?** (Efficient kernel mechanism to watch file system events).

### Practical Challenges

1.  **Implement "Trailer Mode":** Add logic to detect if a trailer is connected (via VHAL) and switch the "Rear" camera to the "Trailer" camera.
2.  **Create a "Night Mode":** If the Ambient Light Sensor (VHAL) reports dark, automatically boost Camera Gain and switch Display to Dark Theme.

---

## 📚 Further Reading & Resources

### Documentation
*   **Android Camera Metadata Tags.**
*   **Linux inotify man page.**

---

## 🎓 Summary

Today we covered:
- ✅ **Parameters:** Controlling the ISP.
- ✅ **Metadata:** Reporting status.
- ✅ **VHAL:** The trigger mechanism.
- ✅ **Hotplug:** Handling USB cameras.
- ✅ **Optimization:** Caching and Reuse.

**Next:** Day 65 - EVS Display & Rendering (OpenGL ES).

---

**Day 64 Complete** | Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS
