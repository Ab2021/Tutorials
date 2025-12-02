# Day 51: Video Encoding Fundamentals (H.264, HEVC, AV1)
## Phase 3: Camera Systems & ISP | Week 9: Video Encoding & Streaming

---

## 🎯 Learning Objectives
1.  **Understand** the principles of Video Compression (Spatial vs Temporal Redundancy).
2.  **Analyze** the structure of a video stream: GOP (Group of Pictures), I/P/B Frames.
3.  **Compare** Codecs: H.264 (AVC), H.265 (HEVC), VP9, AV1.
4.  **Implement** a basic H.264 encoder pipeline using FFmpeg/GStreamer.
5.  **Control** Bitrate: CBR (Constant), VBR (Variable), CQP (Constant Quantization).
6.  **Debug** compression artifacts (Blocking, Ringing, Banding).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** PC or Embedded Board with Hardware Encoder (NVENC, V4L2-M2M).
*   **Software:** FFmpeg, GStreamer.
*   **Knowledge:** DCT (Discrete Cosine Transform), Entropy Coding.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Compress?
*   **Raw 1080p30:** $1920 \times 1080 \times 1.5 (YUV420) \times 30 \approx 93$ MB/s ($750$ Mbps).
*   **Compressed 1080p30:** ~4-8 Mbps.
*   **Ratio:** ~100:1 compression is needed for streaming.

### 🔹 Part 2: How it Works
1.  **Spatial Redundancy (Intra-frame):** Inside one frame, blue sky pixels are similar. Use DCT + Quantization (like JPEG).
2.  **Temporal Redundancy (Inter-frame):** Frame N is similar to Frame N-1. Send only the *difference* (Residual).
    *   **Motion Estimation:** Find where the object moved (Motion Vector).
    *   **Motion Compensation:** Shift the previous frame and subtract.

### 🔹 Part 3: Frame Types (GOP)
*   **I-Frame (Intra):** Full image. Independent. Keyframe. Large size.
*   **P-Frame (Predicted):** References previous frames. Smaller.
*   **B-Frame (Bi-directional):** References past AND future frames. Smallest. High latency (need to wait for future frame).
*   **IDR (Instantaneous Decoder Refresh):** A special I-Frame that clears the reference buffer. Essential for seeking.

### 🔹 Part 4: Codec Evolution
*   **H.264 (AVC):** The industry standard. Good compatibility.
*   **H.265 (HEVC):** ~50% bitrate saving over H.264 for same quality. High CPU usage. Licensing issues.
*   **VP9:** Google's open alternative to HEVC. Used in YouTube.
*   **AV1:** Next-gen open standard. Better than HEVC. Very slow encoding (unless HW accelerated).

---

## 💻 Implementation Examples

### Example 1: FFmpeg Command Line

The universal tool for encoding.

```bash
# Basic H.264 Encoding (Software libx264)
ffmpeg -i input.raw -c:v libx264 -preset fast -crf 23 output.mp4

# Hardware Encoding (Nvidia NVENC)
ffmpeg -i input.raw -c:v h264_nvenc -b:v 5M output.mp4

# Hardware Encoding (Raspberry Pi V4L2)
ffmpeg -i input.raw -c:v h264_v4l2m2m -b:v 5M output.mp4
```

### Example 2: GStreamer Encoding Pipeline (C++)

```cpp
/**
 * @brief Simple GStreamer H.264 Encoder
 */
#include <gst/gst.h>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    
    // Pipeline: Videotestsrc -> x264enc -> MP4Mux -> FileSink
    GstElement *pipeline = gst_parse_launch(
        "videotestsrc ! video/x-raw,width=1920,height=1080,framerate=30/1 ! "
        "x264enc tune=zerolatency bitrate=5000 ! "
        "mp4mux ! filesink location=test.mp4", 
        NULL);
        
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    // Wait for 5 seconds
    g_usleep(5 * 1000000);
    
    // Send EOS (End of Stream) to properly close MP4 file
    gst_element_send_event(pipeline, gst_event_new_eos());
    
    // Wait for EOS to propagate
    GstBus *bus = gst_element_get_bus(pipeline);
    gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_EOS);
    
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return 0;
}
```

### Example 3: Bitrate Control Modes

*   **CBR (Constant Bitrate):** Good for streaming (predictable bandwidth). Quality drops in complex scenes.
    *   `x264enc bitrate=5000`
*   **VBR (Variable Bitrate):** Good for storage. Constant quality, variable size.
    *   `x264enc pass=1` (2-pass encoding)
*   **CQP (Constant Quantization Parameter):** Fixed compression level.
    *   `x264enc quantizer=20` (Lower = Better).

---

## 🔬 Hands-On Lab Exercises

### Lab 1: H.264 vs H.265

**Objective:** Compare efficiency.

**Steps:**
1.  Encode a 1-minute video with H.264 at CRF 23. Record file size.
2.  Encode same video with H.265 (libx265) at CRF 28 (visually similar). Record file size.
3.  **Result:** H.265 file should be 30-50% smaller.
4.  **Cost:** Measure encoding time. H.265 is much slower.

### Lab 2: The "Confetti" Test

**Objective:** Break the encoder.

**Steps:**
1.  Generate a video of random noise or falling confetti (High spatial + High temporal complexity).
2.  Encode at 2 Mbps.
3.  **Observation:** Massive "Macroblocking" artifacts. The encoder runs out of bits and must use huge quantization steps.

### Lab 3: Low Latency Tuning

**Objective:** Optimize for live streaming.

**Steps:**
1.  Default settings: Latency ~500ms (due to B-frames and lookahead).
2.  Apply `tune=zerolatency` (Disables B-frames, Lookahead).
3.  **Result:** Latency drops to < 50ms. Bitrate might increase slightly for same quality.

---

## 🐛 Debugging Techniques

### Debug 1: "Green Smear" / Corruption

**Symptom:** Bottom half of video is green or garbled.

**Cause:**
*   Packet loss in stream.
*   Decoder received a P-frame without the corresponding I-frame.
*   **Fix:** Force an IDR frame (Keyframe) immediately on the encoder side.

### Debug 2: Pulsing Quality

**Symptom:** Quality is good, then bad, then good (every 1 second).

**Cause:**
*   I-Frame pulsing. I-frames consume many bits, starving the following P-frames in CBR mode.
*   **Fix:** Increase VBV buffer size (`vbv-bufsize`) to allow momentary bitrate spikes.

---

## ⚡ Performance Optimization

### Optimization 1: Hardware Acceleration

*   Always use hardware encoders on embedded (Jetson/Pi). CPU encoding 1080p is impossible on ARM cores.
*   `omxh264enc` (Old), `v4l2h264enc` (New standard).

### Optimization 2: ROI Encoding

*   Allocate more bits to the center of the image (or faces).
*   Compress the background heavily.
*   Supported by modern HW encoders (Nvidia, Ambarella).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do B-frames add latency?**
2.  **What is the difference between "Container" (MP4) and "Codec" (H.264)?**
3.  **Why is "Chroma Subsampling" (4:2:0) used?**
4.  **What does "Profile" (Baseline, Main, High) mean in H.264?**

### Practical Challenges

1.  **Implement a "Smart Recorder":** Record in a ring buffer. When an event happens, save the last 10 seconds + next 10 seconds.
2.  **Create a "Transcoder":** Read H.265 file, decode, resize to 720p, encode to H.264.

---

## 📚 Further Reading & Resources

### Standards
*   **ITU-T H.264 Specification.**

### Tools
*   **FFmpeg:** The swiss army knife.
*   **Bitrate Viewer:** Tool to visualize bitrate usage over time.

---

## 🎓 Summary

Today we covered:
- ✅ **Compression:** Spatial vs Temporal.
- ✅ **GOP:** I, P, B frames.
- ✅ **Codecs:** AVC, HEVC, AV1.
- ✅ **Bitrate Control:** CBR vs VBR.
- ✅ **Latency:** Tuning for real-time.

**Next:** Day 52 - Streaming Protocols (RTSP, WebRTC, HLS).

---

**Day 51 Complete** | Phase 3: Camera Systems & ISP | Week 9: Video Encoding & Streaming
