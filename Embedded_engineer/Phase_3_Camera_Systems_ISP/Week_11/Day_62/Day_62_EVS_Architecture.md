# Day 62: EVS (Exterior View System) Architecture
## Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS

---

## 🎯 Learning Objectives
1.  **Understand** the EVS Requirement: < 2 seconds boot-to-video.
2.  **Analyze** the EVS Stack: Kernel -> EVS HAL -> EVS Manager -> EVS App.
3.  **Differentiate** between EVS Camera (Hardware) and Android Camera (Framework).
4.  **Study** the `IEvsCamera` and `IEvsDisplay` HIDL/AIDL interfaces.
5.  **Trace** the flow of a video frame from Sensor to Screen in EVS.
6.  **Debug** EVS startup issues.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** AAOS supported board.
*   **Software:** AOSP Source Code (`packages/services/Car/evs`).
*   **Knowledge:** C++, Binder, HAL.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "2-Second" Rule
*   **Regulation:** FMVSS 111 (USA) requires the backup camera image to be visible within 2 seconds of shifting into Reverse.
*   **Challenge:** Android takes 10-20 seconds to boot.
*   **Solution:** EVS starts *immediately* after the Kernel loads `init`. It does NOT wait for the Java VM (Zygote) or the System Server.

### 🔹 Part 2: EVS Architecture Layers
1.  **Kernel Drivers:** V4L2 drivers for Camera and DRM/KMS drivers for Display.
2.  **EVS HAL (Hardware Abstraction Layer):** A C++ daemon (`android.hardware.automotive.evs@1.1-service`) that talks to Kernel drivers. It implements the HIDL/AIDL interface.
3.  **EVS Manager:** A proxy service that allows multiple clients (EVS App, Android Camera Service) to share the camera.
4.  **EVS Application:** A native C++ app (`evs_app`) that draws the camera frames directly to the display hardware (bypassing SurfaceFlinger initially).

### 🔹 Part 3: Key Interfaces (AIDL)
*   **IEvsEnumerator:** "List available cameras and displays."
*   **IEvsCamera:** "Open camera, start stream, get frames."
*   **IEvsDisplay:** "Get display handle, return target buffer."
*   **IEvsCameraStream:** Callback interface to receive frames.

### 🔹 Part 4: The "Handoff"
1.  **Early Boot:** EVS App starts, grabs the Display, shows RVC.
2.  **Late Boot:** Android SurfaceFlinger starts.
3.  **Handoff:** EVS App releases the Display (or becomes a layer within SurfaceFlinger) so Android UI can appear.

---

## 💻 Implementation Examples

### Example 1: EVS HAL Interface Definition (AIDL)

Simplified view of `IEvsCamera.aidl`.

```java
package android.hardware.automotive.evs;

interface IEvsCamera {
    // Get camera info (Resolution, Format)
    CameraDesc getCameraInfo();

    // Start streaming
    void startVideoStream(in IEvsCameraStream receiver);

    // Stop streaming
    void stopVideoStream();

    // Return a buffer to the HAL (after display is done)
    void doneWithFrame(in BufferDesc buffer);
    
    // Set parameters (Brightness, Contrast)
    void setIntParameter(in CameraParam id, int value);
}
```

### Example 2: EVS Application Logic (C++)

How the native app opens a camera.

```cpp
#include <android/hardware/automotive/evs/1.1/IEvsEnumerator.h>
#include <android/hardware/automotive/evs/1.1/IEvsCamera.h>

using namespace android::hardware::automotive::evs::V1_1;

void start_evs() {
    // 1. Get Enumerator Service
    sp<IEvsEnumerator> pEnum = IEvsEnumerator::getService("default");
    
    // 2. List Cameras
    hidl_vec<CameraDesc> cameras;
    pEnum->getCameraList_1_1([&](auto list) { cameras = list; });
    
    // 3. Open Rear Camera (Assume index 0)
    sp<IEvsCamera> pCamera = pEnum->openCamera_1_1(cameras[0].cameraId, nullptr);
    
    // 4. Start Stream
    pCamera->startVideoStream(new MyStreamHandler());
    
    // 5. Main Loop (Wait for events)
    // ...
}

// Stream Handler Callback
class MyStreamHandler : public IEvsCameraStream {
    Return<void> deliverFrame(const BufferDesc& buffer) override {
        // Render buffer to Display via OpenGL/Vulkan
        render_frame(buffer);
        
        // Return buffer to HAL
        pCamera->doneWithFrame(buffer);
        return Void();
    }
};
```

### Example 3: `init.rc` Configuration

Ensuring EVS starts early.

```bash
# /vendor/etc/init/evs_app.rc

service evs_app /vendor/bin/evs_app
    class core             # Start in 'core' class (very early)
    user automotive_evs
    group automotive_evs
    priority -20           # Highest priority
    disabled               # Started by VHAL trigger usually
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Browsing the EVS Source

**Objective:** Locate the code.

**Steps:**
1.  Navigate to `packages/services/Car/evs`.
2.  Explore `manager/` (The proxy).
3.  Explore `sampleDriver/` (A sample HAL implementation).
4.  Explore `app/` (The reference EVS app).

### Lab 2: Simulating EVS Start

**Objective:** Run the app manually.

**Steps:**
1.  On Cuttlefish/Emulator.
2.  Stop the Android UI: `adb shell stop`.
3.  Run EVS App: `adb shell /vendor/bin/evs_app`.
4.  **Observation:** You should see the camera feed (or a test pattern) on the screen, even though the Android UI is dead.

### Lab 3: Analyzing EVS Logs

**Objective:** Trace the startup.

**Steps:**
1.  `adb logcat -s EVS`.
2.  Look for:
    *   `EVS App starting`
    *   `Requesting Display`
    *   `Camera Stream Started`
3.  **Metric:** Check the timestamp difference between `init` start and `Camera Stream Started`.

---

## 🐛 Debugging Techniques

### Debug 1: "Display Busy"

**Symptom:** EVS App crashes with "Failed to open display".

**Cause:**
*   SurfaceFlinger has already grabbed the DRM Master lock.
*   **Fix:** Ensure EVS starts *before* SurfaceFlinger, or configure EVS Manager to act as a SurfaceFlinger client (Late Boot mode).

### Debug 2: Frame Drop / Latency

**Symptom:** Video is choppy.

**Cause:**
*   HAL is allocating new buffers every frame (Slow).
*   **Fix:** Use a Buffer Pool. Allocate 3-4 buffers at start and cycle them.

---

## ⚡ Performance Optimization

### Optimization 1: Zero-Copy Rendering

*   The Camera HAL outputs a `HardwareBuffer` (DMA-BUF).
*   The EVS App imports this buffer as an EGL Image (OpenGL texture).
*   **Result:** No `memcpy` needed. The GPU reads directly from the Camera buffer.

### Optimization 2: Direct Mode

*   Bypass EVS Manager if only one client exists.
*   The EVS App talks directly to EVS HAL.
*   Reduces IPC (Inter-Process Communication) overhead.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does EVS need to bypass SurfaceFlinger in early boot?** (SurfaceFlinger takes too long to load).
2.  **What is the role of the EVS Manager?** (Multiplexing: Allows RVC and Dashcam to use the camera simultaneously).
3.  **How does the EVS App know when to stop?** (VHAL sends a "Gear Drive" event).
4.  **What is "PMEM" or "ION/DMA-BUF"?** (Shared memory mechanisms for zero-copy).

### Practical Challenges

1.  **Modify EVS App:** Change the reference app to display a static "WARNING: CHECK SURROUNDINGS" text overlay on the video.
2.  **Measure Boot Time:** Use `bootchart` to visualize exactly when `evs_app` starts relative to other services.

---

## 📚 Further Reading & Resources

### Documentation
*   **Android EVS HAL Interface Specification.**
*   **Khronos EGL Extension for Android Native Fence.**

---

## 🎓 Summary

Today we covered:
- ✅ **Requirement:** 2-second boot.
- ✅ **Architecture:** Kernel -> HAL -> Manager -> App.
- ✅ **Interfaces:** IEvsCamera, IEvsDisplay.
- ✅ **Flow:** Zero-copy streaming.
- ✅ **Handoff:** Coexisting with Android UI.

**Next:** Day 63 - EVS HAL Implementation (Writing the Driver Wrapper).

---

**Day 62 Complete** | Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS
