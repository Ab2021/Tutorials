# Day 45: Week 7 Review - Edge AI Project
## Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI

---

## 🎯 Learning Objectives
1.  **Integrate** Camera Capture, Pre-processing, Inference, and Display into a single pipeline.
2.  **Deploy** a YOLO object detector on an embedded device (Jetson/Pi).
3.  **Implement** logic to trigger actions based on detections (e.g., "Person Detected" -> Save Video).
4.  **Optimize** the pipeline for real-time performance (30 FPS).
5.  **Validate** the system in real-world conditions (Day/Night, Distance).

---

## 📚 Week 7 Recap

### Topics Covered

**Day 40: CV Basics**
- OpenCV, Mat, Color Spaces (HSV), Morphology.

**Day 41: Feature Tracking**
- Harris/FAST Corners, ORB, Optical Flow (Lucas-Kanade).

**Day 42: Classical ML Detection**
- Haar Cascades (Face), HOG (Pedestrian).

**Day 43: Deep Learning Vision**
- CNNs, YOLO, SSD, OpenCV DNN.

**Day 44: Edge Acceleration**
- TensorRT, TFLite, Quantization (INT8).

---

## 💻 Week 7 Integration Project

### Project: "SentryCam - Smart Security System"

**Objective:** Build a C++ application that captures video, detects "Person" and "Car", draws bounding boxes, and saves a video clip when a person enters a specific "Danger Zone".

**Features:**
1.  **Capture:** Libcamera / V4L2 (1080p30).
2.  **Inference:** YOLOv4-Tiny (TensorRT optimized).
3.  **Logic:** Polygon-based Zone intrusion detection.
4.  **Output:** RTSP Stream (GStreamer) + Local Recording.

### Architecture

```mermaid
graph LR
    CAM[Camera Source] --> PRE[Pre-Process (Resize/Norm)]
    PRE --> INFER[TensorRT Inference]
    INFER --> POST[NMS & Parsing]
    POST --> LOGIC[Zone Logic]
    LOGIC --> OSD[Draw Boxes]
    OSD --> DISP[Display/RTSP]
    LOGIC --> REC[Video Recorder]
```

### Implementation

#### Part 1: The Zone Logic

```cpp
struct Point { int x, y; };

/**
 * @brief Check if point is inside polygon (Ray Casting algo)
 */
bool is_inside_zone(Point p, const std::vector<Point>& zone) {
    int i, j, c = 0;
    for (i = 0, j = zone.size()-1; i < zone.size(); j = i++) {
        if ( ((zone[i].y > p.y) != (zone[j].y > p.y)) &&
             (p.x < (zone[j].x - zone[i].x) * (p.y - zone[i].y) / (double)(zone[j].y - zone[i].y) + zone[i].x) )
           c = !c;
    }
    return c;
}

void process_detections(const std::vector<Detection>& dets, AppState& state) {
    bool person_in_zone = false;
    
    for (const auto& d : dets) {
        if (d.class_id == CLASS_PERSON) {
            Point center = { d.bbox.x + d.bbox.width/2, d.bbox.y + d.bbox.height/2 };
            
            if (is_inside_zone(center, state.danger_zone)) {
                person_in_zone = true;
                // Draw Red Box
            } else {
                // Draw Green Box
            }
        }
    }
    
    if (person_in_zone && !state.recording) {
        start_recording();
    } else if (!person_in_zone && state.recording) {
        stop_recording(); // with delay
    }
}
```

#### Part 2: The Pipeline (Pseudo-Code)

```cpp
void main_loop() {
    // 1. Init
    Camera cam; cam.open();
    YoloTRT yolo("yolov4-tiny.engine");
    VideoWriter writer;
    
    // 2. Loop
    while(true) {
        Mat frame = cam.capture();
        
        // Async Inference (Pipelining)
        // If we have a previous future, get result
        // Start new inference
        
        auto dets = yolo.detect(frame);
        
        // Logic
        process_detections(dets, state);
        
        // OSD
        draw_osd(frame, dets);
        
        // Recording
        if (state.recording) {
            writer.write(frame);
        }
        
        // Display
        imshow("SentryCam", frame);
        if (waitKey(1) == 27) break;
    }
}
```

---

## 🔬 System Validation Plan

### Test 1: Latency Benchmark
**Objective:** Measure glass-to-glass latency.
**Procedure:**
1.  Film a stopwatch with the SentryCam.
2.  Display the SentryCam output on a monitor.
3.  Take a photo of both the real stopwatch and the monitor.
4.  **Result:** Difference = Latency. Target < 100ms.

### Test 2: Detection Range
**Objective:** How far can it see?
**Procedure:**
1.  Person walks away from camera.
2.  Mark the distance where detection flickers or fails.
3.  **Goal:** 10 meters for YOLO-Tiny (depends on lens FoV).

### Test 3: False Alarm Rate (Night)
**Objective:** Test robustness.
**Procedure:**
1.  Leave running overnight.
2.  Count recordings triggered by moths, car headlights, or noise.
3.  **Fix:** Increase confidence threshold or use "Motion History" to filter transient noise.

---

## 🐛 Troubleshooting Guide

### Issue 1: Overheating

**Symptom:** FPS drops after 10 minutes.

**Cause:**
*   Thermal throttling on Jetson/Pi.
*   **Fix:** Add a fan. Use `jtop` to monitor thermals. Reduce clock speed if necessary.

### Issue 2: Memory Leak

**Symptom:** App crashes after 1 hour (OOM).

**Cause:**
*   OpenCV `Mat` not released (rare in C++ RAII, but possible with pointers).
*   GStreamer pipeline leaking buffers.
*   **Fix:** Run with `valgrind`. Check GStreamer bus messages.

---

## 📝 Assessment Questions

### Comprehensive Questions

1.  **Design a "Privacy Mask" feature:** How would you blur faces *before* saving the video?
2.  **Explain the trade-off between "Input Resolution" and "Detection Range".**
3.  **How does "Batching" affect latency vs throughput in this specific project?** (Hint: We process 1 frame at a time, so batching adds pure latency).

### Practical Challenges

1.  **Add "Loitering Detection":** Trigger alarm only if a person stays in the zone for > 5 seconds. (Requires Object Tracking ID).
2.  **Implement MQTT Alert:** Publish a message `{"alert": "person", "time": 12345}` to a broker when detection occurs.

---

## 📚 Resources & Next Steps

### Week 7 Summary

**Completed:**
- ✅ **CV Basics:** The foundation.
- ✅ **Tracking:** Following pixels.
- ✅ **Detection:** Finding objects.
- ✅ **Acceleration:** Making it fast.
- ✅ **Project:** Real-world application.

**Key Skills Acquired:**
- Integrating AI into C++ pipelines.
- Optimizing for Edge Hardware.
- Handling real-time video streams.

### Week 8 Preview (Phase 3 Continued)

**Topics:**
- **Stereo Vision:** Depth perception.
- **3D Reconstruction:** Point clouds.
- **SLAM:** Simultaneous Localization and Mapping.
- **Structure from Motion (SfM).**

---

**Day 45 Complete** | Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI
