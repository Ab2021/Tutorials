# Day 63: EVS HAL Implementation
## Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS

---

## 🎯 Learning Objectives
1.  **Implement** a basic EVS HAL Service (`android.hardware.automotive.evs@1.1-service`).
2.  **Wrap** a V4L2 camera driver into an `IEvsCamera` object.
3.  **Manage** Gralloc buffers for video frames.
4.  **Handle** multiple clients via the EVS Manager.
5.  **Implement** the `getCameraList` and `openCamera` methods.
6.  **Debug** HAL crashes and binder transaction failures.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Linux system with a V4L2 camera (`/dev/video0`).
*   **Software:** AOSP build system (`mm`, `m`).
*   **Knowledge:** C++, V4L2 ioctls, Android HIDL/AIDL.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The HAL Service Structure
An EVS HAL is a standalone process (daemon) that registers itself with the `hwservicemanager`.
*   **Entry Point:** `service.cpp` (Main function).
*   **Enumerator:** `EvsEnumerator.cpp` (Implements `IEvsEnumerator`).
*   **Camera:** `EvsCamera.cpp` (Implements `IEvsCamera`).
*   **Display:** `EvsDisplay.cpp` (Implements `IEvsDisplay`).

### 🔹 Part 2: Buffer Management (Gralloc)
*   EVS cannot use `malloc`. It must use **Gralloc** (Graphics Allocator) to create buffers that can be shared with the GPU (for display) and the Encoder (for recording).
*   **AHardwareBuffer:** The Native Android type for these buffers.
*   **Usage Flags:** `GRALLOC_USAGE_HW_CAMERA_WRITE`, `GRALLOC_USAGE_HW_TEXTURE`.

### 🔹 Part 3: Threading Model
*   **Main Thread:** Handles Binder calls (Open, Close, Start).
*   **Capture Thread:** A dedicated thread that loops on `ioctl(VIDIOC_DQBUF)`, wraps the V4L2 buffer into a HAL buffer, and calls `deliverFrame`.

---

## 💻 Implementation Examples

### Example 1: `EvsEnumerator::getCameraList`

Returning the list of available cameras.

```cpp
Return<void> EvsEnumerator::getCameraList_1_1(getCameraList_1_1_cb _hidl_cb) {
    hidl_vec<CameraDesc> cameras;
    cameras.resize(1);
    
    // Define Camera 0 (Rear View)
    cameras[0].cameraId = "v4l2_camera_0";
    cameras[0].metadata.resize(0); // Add static metadata here if needed
    
    // Return the list
    _hidl_cb(cameras);
    return Void();
}
```

### Example 2: `EvsCamera::startVideoStream`

Setting up the capture loop.

```cpp
Return<EvsResult> EvsCamera::startVideoStream(const sp<IEvsCameraStream>& receiver) {
    std::lock_guard<std::mutex> lock(mAccessLock);
    
    if (mStreamState == RUNNING) return EvsResult::STREAM_ALREADY_RUNNING;
    
    mReceiver = receiver;
    
    // Open V4L2 Device
    mVideoFd = open("/dev/video0", O_RDWR);
    
    // Configure Format (YUYV 1280x720)
    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 1280;
    fmt.fmt.pix.height = 720;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    ioctl(mVideoFd, VIDIOC_S_FMT, &fmt);
    
    // Request Buffers
    // ... (Standard V4L2 REQBUFS / QUERYBUF / QBUF)
    
    // Start Capture Thread
    mCaptureThread = std::thread([this](){
        while (mRunThread) {
            // 1. Dequeue V4L2 Buffer
            struct v4l2_buffer buf = {};
            ioctl(mVideoFd, VIDIOC_DQBUF, &buf);
            
            // 2. Wrap in BufferDesc
            BufferDesc evsBuf;
            evsBuf.bufferId = buf.index;
            evsBuf.memHandle = mGrallocHandles[buf.index]; // Pre-allocated
            
            // 3. Deliver
            if (mReceiver) {
                mReceiver->deliverFrame(evsBuf);
            }
            
            // 4. Wait for return (In real HAL, we wait for doneWithFrame)
            // For simplicity, we assume immediate return or use a pool
        }
    });
    
    mStreamState = RUNNING;
    return EvsResult::OK;
}
```

### Example 3: `doneWithFrame`

Recycling buffers.

```cpp
Return<void> EvsCamera::doneWithFrame(const BufferDesc& buffer) {
    // Queue buffer back to V4L2 driver
    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP; // or DMABUF
    buf.index = buffer.bufferId;
    
    ioctl(mVideoFd, VIDIOC_QBUF, &buf);
    return Void();
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Building the Sample HAL

**Objective:** Compile the provided sample.

**Steps:**
1.  Go to `packages/services/Car/evs/sampleDriver`.
2.  Run `mm`.
3.  **Result:** `android.hardware.automotive.evs@1.1-sample.so` and the executable service.
4.  Push to device: `adb push ... /vendor/bin/hw/`.

### Lab 2: Mocking a Camera

**Objective:** Test without hardware.

**Steps:**
1.  Modify `EvsCamera.cpp` to ignore V4L2.
2.  In the capture thread, generate a synthetic image (e.g., a moving color bar).
3.  Use `usleep(33000)` to simulate 30fps.
4.  **Verify:** Run EVS App. You should see the color bars.

### Lab 3: Stress Test

**Objective:** Check for memory leaks.

**Steps:**
1.  Start EVS.
2.  Stop EVS.
3.  Repeat 100 times script.
4.  **Monitor:** `adb shell dumpsys meminfo android.hardware.automotive.evs...`.
5.  **Goal:** Memory usage should remain stable. If it grows, you are leaking Gralloc buffers.

---

## 🐛 Debugging Techniques

### Debug 1: SELinux Denials

**Symptom:** HAL crashes or fails to open `/dev/video0`.

**Cause:**
*   SELinux policy prevents the `evs_driver` domain from accessing `video_device`.
*   **Fix:** Check `dmesg | grep avc`. Add rules to `evs_driver.te`.
    ```text
    allow evs_driver video_device:chr_file { open read write ioctl };
    ```

### Debug 2: Buffer Format Mismatch

**Symptom:** Garbled video (Green/Pink lines).

**Cause:**
*   V4L2 outputs YUYV (YUV422), but EVS App expects NV21 or RGBA.
*   **Fix:** Ensure the HAL converts the format OR configures the V4L2 device to output what the consumer expects. Ideally, use the ISP to output NV12/NV21 directly.

---

## ⚡ Performance Optimization

### Optimization 1: DMA-BUF Import

*   Instead of `mmap`ing the V4L2 buffer and copying it to a Gralloc buffer (Slow!), use `VIDIOC_EXPBUF` to export the V4L2 buffer as a file descriptor (dmabuf_fd).
*   Import this FD into a Gralloc handle.
*   **Result:** Zero-copy from Sensor to GPU.

### Optimization 2: High Priority Thread

*   Set the capture thread priority to `SCHED_FIFO` or `SCHED_RR` (Real-time).
*   Prevents glitches when the system is under load.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do we need `doneWithFrame`?** (To tell the producer it can overwrite the memory).
2.  **What is the difference between HIDL and AIDL?** (Legacy vs New IPC mechanism).
3.  **Why is `ioctl` blocking?** (It waits for the hardware interrupt).
4.  **What is a "Fence" in Android graphics?** (A synchronization primitive to wait for GPU/Hardware completion).

### Practical Challenges

1.  **Implement "Frame Counter":** Draw a frame number on the image in the HAL (using CPU write) to verify latency/drops.
2.  **Multi-Camera Support:** Extend the Enumerator to return 4 cameras (Front, Back, Left, Right).

---

## 📚 Further Reading & Resources

### Documentation
*   **V4L2 API Specification.**
*   **Android Gralloc Header Files.**

---

## 🎓 Summary

Today we covered:
- ✅ **HAL Structure:** Enumerator, Camera, Display.
- ✅ **V4L2 Integration:** Wrapping ioctls.
- ✅ **Buffer Management:** Gralloc and BufferDesc.
- ✅ **Concurrency:** Capture threads.
- ✅ **Zero-Copy:** The holy grail of performance.

**Next:** Day 64 - EVS Camera Integration (Advanced Features & Metadata).

---

**Day 63 Complete** | Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS
