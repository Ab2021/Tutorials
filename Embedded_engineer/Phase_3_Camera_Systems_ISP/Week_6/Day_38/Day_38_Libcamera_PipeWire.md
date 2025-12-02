# Day 38: Libcamera & PipeWire (Linux Camera Stack)
## Phase 3: Camera Systems & ISP | Week 6: Image Quality & Tuning

---

## 🎯 Learning Objectives
1.  **Understand** why V4L2 is insufficient for modern complex cameras (Graph complexity, 3A).
2.  **Analyze** the Libcamera architecture (Camera Manager, Pipeline Handler, IPA).
3.  **Master** the PipeWire multimedia bus for sharing camera streams.
4.  **Implement** a simple Libcamera application (Capture & Preview).
5.  **Configure** Image Processing Algorithms (IPA) in Libcamera.
6.  **Debug** Libcamera pipelines using `cam` and `qcam`.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Raspberry Pi (with Libcamera support) or Linux PC.
*   **Software:** `libcamera`, `pipewire`, `gstreamer`.
*   **Knowledge:** C++, V4L2 (Legacy).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem with V4L2
*   **Complexity:** Modern ISPs have complex topologies (Resizers, Stats, Multiple Streams). Exposing this via Media Controller is too hard for generic apps.
*   **No 3A:** V4L2 has no standard for Auto-Exposure/White Balance algorithms (User-space usually handles it ad-hoc).
*   **Solution:** **Libcamera** provides a unified userspace API that abstracts the complex Media Controller graph and handles 3A.

### 🔹 Part 2: Libcamera Architecture
1.  **Camera Manager:** Enumerates devices.
2.  **Pipeline Handler:** Device-specific code (e.g., for Raspberry Pi VC4, IPU3, RKISP1). Sets up the Media Graph.
3.  **IPA (Image Processing Algorithms):** Sandboxed module running AE/AWB/AF. It receives stats and computes controls.
4.  **Application API:** Clean C++ API for Request/Stream management (similar to Android HAL3).

### 🔹 Part 3: PipeWire
*   **Role:** The "PulseAudio for Video".
*   **Function:** Allows multiple applications to share the camera (e.g., Browser + OBS).
*   **Integration:** Libcamera plugin for PipeWire makes Libcamera devices visible to all PipeWire apps (Firefox, Chrome).

---

## 💻 Implementation Examples

### Example 1: Basic Libcamera Capture (C++)

```cpp
/**
 * @file simple_cam.cpp
 * @brief Capture one frame using Libcamera
 */

#include <libcamera/libcamera.h>
#include <iostream>
#include <memory>

using namespace libcamera;

int main() {
    // 1. Start Camera Manager
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
    cm->start();
    
    // 2. Get First Camera
    if (cm->cameras().empty()) {
        std::cerr << "No cameras found" << std::endl;
        return -1;
    }
    std::shared_ptr<Camera> camera = cm->cameras()[0];
    camera->acquire();
    
    // 3. Configure Stream
    std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({ StreamRole::StillCapture });
    StreamConfiguration &streamConfig = config->at(0);
    streamConfig.pixelFormat = formats::MJPEG;
    streamConfig.size = {640, 480};
    config->validate();
    camera->configure(config.get());
    
    // 4. Allocate Buffers
    FrameBufferAllocator *allocator = new FrameBufferAllocator(camera);
    allocator->allocate(streamConfig.stream());
    
    // 5. Create Request
    std::unique_ptr<Request> request = camera->createRequest();
    const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(streamConfig.stream());
    request->addBuffer(streamConfig.stream(), buffers[0].get());
    
    // 6. Start & Queue
    camera->start();
    camera->queueRequest(request.get());
    
    // (Wait for completion - simplified)
    std::cout << "Request Queued. Waiting..." << std::endl;
    sleep(1);
    
    // 7. Cleanup
    camera->stop();
    camera->release();
    cm->stop();
    
    return 0;
}
```

### Example 2: PipeWire GStreamer Pipeline

Using PipeWire to access a Libcamera device in GStreamer.

```bash
# List PipeWire sources
pw-cli list-objects | grep "Camera"

# Play stream using pipewiresrc
gst-launch-1.0 pipewiresrc ! videoconvert ! autovideosink
```

### Example 3: IPA Configuration (Tuning)

Libcamera loads tuning files (JSON) for the IPA.

```json
/* imx219.json (Raspberry Pi Tuning) */
{
    "algorithms": [
        {
            "name": "BlackLevel",
            "black_level": 4096
        },
        {
            "name": "Awb",
            "mode": "GreyWorld",
            "cadence": 5
        },
        {
            "name": "Agc",
            "exposure_mode": "Normal",
            "constraint_mode": "Normal"
        }
    ]
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Using `cam` and `qcam`

**Objective:** Test Libcamera tools.

**Steps:**
1.  Run `cam -l` to list cameras.
2.  Run `qcam` (Qt GUI).
3.  **Observation:** You should see a live preview.
4.  **Experiment:** Adjust AE/AWB controls in the GUI. Observe the response latency.

### Lab 2: GStreamer Integration (`libcamerasrc`)

**Objective:** Stream to network.

**Steps:**
1.  Install `gstreamer1.0-libcamera`.
2.  Run:
    ```bash
    gst-launch-1.0 libcamerasrc ! video/x-raw,width=1280,height=720 ! videoconvert ! autovideosink
    ```
3.  **Challenge:** Pipe it to H.264 encoder and save to file.

### Lab 3: Writing a Custom IPA Module

**Objective:** Add a simple "Green Tint" effect.

**Steps:**
1.  Modify the IPA source code (e.g., `src/ipa/raspberrypi/raspberrypi.cpp`).
2.  In the AWB algorithm, force `Green Gain = 2.0`.
3.  Recompile Libcamera.
4.  Run `qcam`.
5.  **Result:** The image should look Matrix-green.

---

## 🐛 Debugging Techniques

### Debug 1: "Pipeline Handler Not Found"

**Symptom:** `cam -l` returns empty.

**Cause:**
*   Kernel drivers not loaded.
*   Device Tree mismatch.
*   Libcamera compiled without support for your platform (e.g., missing `-Dpipelines=raspberrypi`).
*   **Fix:** Check `dmesg` for V4L2 devices. Recompile Libcamera with correct flags.

### Debug 2: IPA Error / Sandbox Crash

**Symptom:** "IPA IPC failure".

**Cause:**
*   IPA signature mismatch (security feature).
*   Tuning file parsing error.
*   **Fix:** Run with `LIBCAMERA_LOG_LEVELS=*:DEBUG` to see detailed logs.

---

## ⚡ Performance Optimization

### Optimization 1: Zero-Copy Buffers

*   Libcamera supports passing `dmabuf` FDs directly to the display (DRM/KMS) or Encoder.
*   Avoid `mmap` and CPU copy at all costs for 4K video.

### Optimization 2: Multi-Stream Configuration

*   Configure "Viewfinder" (Low Res) and "Main" (High Res) streams simultaneously.
*   The ISP hardware (if supported) will generate both from a single sensor read, saving bandwidth.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does Libcamera run IPA in a separate process (Sandbox)?**
2.  **How does PipeWire differ from PulseAudio?**
3.  **What is the role of the "Pipeline Handler"?**
4.  **Why is `libcamerasrc` preferred over `v4l2src` for complex cameras?**

### Practical Challenges

1.  **Create a "Surveillance App":** Use Libcamera to detect motion (diff between frames) and save a JPEG when motion is detected.
2.  **Build a WebRTC Streamer:** Connect Libcamera -> PipeWire -> Web Browser (WebRTC) to stream video to another PC.

---

## 📚 Further Reading & Resources

### Documentation
*   **libcamera.org:** Official documentation and tutorials.
*   **pipewire.org:** Architecture overview.

### Source Code
*   **git.libcamera.org:** Browse the `src/libcamera/pipeline` directory to see how different SoCs are supported.

---

## 🎓 Summary

Today we covered:
- ✅ **Libcamera:** The new standard for Linux cameras.
- ✅ **Architecture:** Manager, Pipeline, IPA.
- ✅ **PipeWire:** Sharing the camera.
- ✅ **Tools:** `cam`, `qcam`, `libcamerasrc`.
- ✅ **Tuning:** JSON-based IPA configuration.

**Next:** Day 39 - Week 6 Review & Driver Project.

---

**Day 38 Complete** | Phase 3: Camera Systems & ISP | Week 6: Image Quality & Tuning
