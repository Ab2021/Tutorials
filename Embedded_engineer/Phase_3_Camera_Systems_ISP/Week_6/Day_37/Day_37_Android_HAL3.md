# Day 37: Android Camera HAL (HAL3) Overview
## Phase 3: Camera Systems & ISP | Week 6: Image Quality & Tuning

---

## 🎯 Learning Objectives
1.  **Understand** the Android Camera Architecture (App -> Framework -> Service -> HAL -> Kernel).
2.  **Analyze** the Camera HAL3 Interface (HIDL/AIDL).
3.  **Master** the Request/Result model (Per-frame control).
4.  **Configure** Stream Combinations (Preview, Video, Snapshot, ZSL).
5.  **Implement** a basic HAL module skeleton.
6.  **Debug** HAL issues using `dumpsys media.camera`.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Android Device (Pixel/Dev Board) or Emulator.
*   **Software:** AOSP Source Code (optional but recommended), Android Studio.
*   **Knowledge:** C++, Binder IPC, V4L2.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Android Camera Stack
1.  **Application:** Uses `Camera2` API (Java/Kotlin).
2.  **Framework:** `CameraService` (C++) handles arbitration and buffering.
3.  **HAL (Hardware Abstraction Layer):** The vendor-specific library that talks to the hardware.
4.  **Kernel:** V4L2 drivers (Sensor, ISP).

### 🔹 Part 2: The HAL3 Model
Unlike V4L2 (Stateful), HAL3 is **Stateless** (mostly).
*   **Request:** Contains ALL settings for a specific frame (Exposure, Gain, Focus Pos, Output Buffers).
*   **Result:** Contains the Metadata (what actually happened) and the filled Image Buffers.
*   **Pipeline:** The Framework sends a stream of Requests. The HAL returns a stream of Results.

### 🔹 Part 3: Streams
The HAL doesn't just output one image. It outputs multiple "Streams" simultaneously from the same sensor data.
*   **Preview:** 1080p, 30fps, YUV (for display).
*   **Video:** 4K, 30fps, YUV (for encoder).
*   **Snapshot:** 12MP, JPEG (on demand).
*   **ZSL (Zero Shutter Lag):** Full-res YUV ring buffer.

---

## 💻 Implementation Examples

### Example 1: HAL Interface Definition (AIDL/HIDL)

The interface the HAL must implement.

```cpp
/* ICameraDeviceSession.hal (Simplified) */

interface ICameraDeviceSession {
    /**
     * Submit a list of capture requests.
     */
    processCaptureRequest(vec<CaptureRequest> requests) generates (Status status, uint32_t numRequestProcessed);

    /**
     * Configure streams (resolution, format).
     */
    configureStreams(StreamConfiguration requestedConfiguration) generates (Status status, HalStreamConfiguration halConfiguration);
    
    /**
     * Flush all pending requests.
     */
    flush() generates (Status status);
};
```

### Example 2: Processing a Request (C++)

Inside the HAL implementation.

```cpp
/**
 * @brief Process a Capture Request
 */
Return<void> CameraDeviceSession::processCaptureRequest(const vec<CaptureRequest>& requests, processCaptureRequest_cb _hidl_cb) {
    
    for (const auto& req : requests) {
        // 1. Parse Settings (Metadata)
        // e.g., Extract Exposure Time
        int64_t exposure_time = get_entry(req.settings, ANDROID_SENSOR_EXPOSURE_TIME);
        
        // 2. Configure ISP/Sensor
        // Apply settings to hardware (V4L2 controls)
        v4l2_set_ctrl(V4L2_CID_EXPOSURE, exposure_time);
        
        // 3. Dequeue Buffer from Stream
        // Get a V4L2 buffer
        
        // 4. Capture
        // Wait for hardware to fill buffer
        
        // 5. Return Result
        CaptureResult result;
        result.frameNumber = req.frameNumber;
        result.resultMetadata = req.settings; // Update with actuals
        result.outputBuffers = req.outputBuffers; // Filled buffers
        
        // Send callback to Framework
        mCallback->processCaptureResult(result);
    }
    
    _hidl_cb(Status::OK, requests.size());
    return Void();
}
```

### Example 3: Metadata Handling (CameraMetadata)

Android uses a specialized dictionary for settings.

```cpp
#include <camera/CameraMetadata.h>

void setup_static_metadata(CameraMetadata& meta) {
    // Advertise capabilities
    
    // Available Resolutions
    int32_t stream_configs[] = {
        HAL_PIXEL_FORMAT_IMPLEMENTATION_DEFINED, 1920, 1080, ANDROID_SCALER_AVAILABLE_STREAM_CONFIGURATIONS_OUTPUT,
        HAL_PIXEL_FORMAT_BLOB, 4000, 3000, ANDROID_SCALER_AVAILABLE_STREAM_CONFIGURATIONS_OUTPUT
    };
    meta.update(ANDROID_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, stream_configs, 8);
    
    // Exposure Range
    int64_t exposure_range[] = {10000, 1000000000}; // 10us to 1s
    meta.update(ANDROID_SENSOR_INFO_EXPOSURE_TIME_RANGE, exposure_range, 2);
    
    // Hardware Level
    uint8_t hardware_level = ANDROID_INFO_SUPPORTED_HARDWARE_LEVEL_FULL;
    meta.update(ANDROID_INFO_SUPPORTED_HARDWARE_LEVEL, &hardware_level, 1);
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Inspecting Camera Characteristics

**Objective:** See what your phone supports.

**Steps:**
1.  Connect Android phone via ADB.
2.  Run `adb shell dumpsys media.camera`.
3.  **Analyze Output:**
    *   Look for `CameraDeviceClient`.
    *   Check `Stream configuration`.
    *   Check `Request/Result` history.
4.  **Observation:** You can see the exact frame rate, resolution, and active streams.

### Lab 2: CTS (Compatibility Test Suite)

**Objective:** Verify HAL compliance.

**Steps:**
1.  Download Android CTS.
2.  Run `run cts -m CtsCameraTestCases`.
3.  **Observation:** Thousands of tests will run (Exposure, Focus, Flash, Multi-stream).
4.  **Failure:** If your HAL crashes or returns wrong metadata, CTS fails. This is the "Gatekeeper" for GMS certification.

### Lab 3: Raw Capture via Camera2 API

**Objective:** Capture Raw from App.

**Steps:**
1.  Write a simple Android App.
2.  Create a `CameraCaptureSession`.
3.  Add a target: `ImageReader` with format `ImageFormat.RAW_SENSOR`.
4.  Capture.
5.  **Result:** You get a `.dng` file (Digital Negative) containing the raw pixel data and metadata (CCM, Black Level) populated by the HAL.

---

## 🐛 Debugging Techniques

### Debug 1: "Camera Error 1" (Server Died)

**Symptom:** Camera app crashes, "Can't connect to camera".

**Cause:**
*   HAL crashed (Segfault).
*   HAL timed out (Watchdog).
*   **Fix:** Check `adb logcat | grep -i camera`. Look for "FATAL" or "tombstone".

### Debug 2: Preview Freeze

**Symptom:** App UI is responsive, but viewfinder is stuck.

**Cause:**
*   HAL is not returning Results.
*   HAL is not returning Buffers.
*   **Fix:** Check if the ISP interrupt is firing. Check if `processCaptureResult` is being called.

---

## ⚡ Performance Optimization

### Optimization 1: Zero Copy

*   Pass buffers from V4L2 (Kernel) to HAL to Framework to GPU (SurfaceFlinger) without `memcpy`.
*   Use `dmabuf` (ION/Gralloc) handles.

### Optimization 2: Request Pipelining

*   Don't wait for Frame N to finish before submitting Frame N+1.
*   Keep the ISP pipeline full (Queue depth > 3).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is HAL3 considered "Stateless"?**
2.  **What is the difference between `LIMITED`, `FULL`, and `LEVEL_3` hardware levels?**
3.  **How does the Framework know if the HAL supports 4K video?**
4.  **What is a "Reprocess Request" (used in ZSL)?**

### Practical Challenges

1.  **Implement a "Fake HAL":** A HAL that returns a generated pattern (Color Bars) instead of real sensor data. Useful for framework testing.
2.  **Trace a Request:** Draw a sequence diagram showing a Request traveling from App -> Framework -> HAL -> Kernel and the Result traveling back.

---

## 📚 Further Reading & Resources

### Documentation
*   **source.android.com:** "Camera HAL3" section.
*   **Camera2 API:** Developer documentation.

### Code
*   **`hardware/libhardware/modules/camera`:** Reference HAL implementation.
*   **`frameworks/av/services/camera/libcameraservice`:** The Framework service.

---

## 🎓 Summary

Today we covered:
- ✅ **Android Camera Stack:** The big picture.
- ✅ **HAL3 Interface:** Requests and Results.
- ✅ **Streams:** Managing multiple outputs.
- ✅ **Metadata:** The language of settings.
- ✅ **CTS:** The validation standard.

**Next:** Day 38 - Libcamera & PipeWire (Linux Camera Stack).

---

**Day 37 Complete** | Phase 3: Camera Systems & ISP | Week 6: Image Quality & Tuning
