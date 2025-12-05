# Day 35: Week 5 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 5: Edge AI & Optimization

---

> **📝 Content Creator Instructions:**
> We have trained massive models and built complex graphs. Now we make them *fast*.
> - **Goal:** Integrate TensorRT, Zero-Copy ROS 2, and CUDA into a single pipeline.
> - **Code:** "The Speed Demon" - A 50 FPS Detection/Tracking stack on Jetson Nano.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Analyze** the "Critical Path" of a robotics pipeline to identify bottlenecks (CPU vs GPU vs Memory).
2.  **Combine** multiple optimization techniques: Quantization + Zero-Copy + RTOS Scheduling.
3.  **Construct** an architecture where Perception (Heavy) does not block Control (Real-Time).
4.  **Validate** system latency and throughput requirements.

---

## 📚 Week 5 Review: The Optimization Pyramid

| Day | Topic | Technique | Benefit | Trade-off |
|-----|-------|-----------|---------|-----------|
| **29** | **Compression** | INT8 Quantization | 4x Speed, 4x Memory | Slight Accuracy Loss |
| **30** | **TensorRT** | Fusion, Auto-Tuning | Max GPU Throughput | Long compilation, Hardware Specific |
| **31** | **Hardware** | DLA / TPU | Efficient Compute | Rigid, limited ops |
| **32** | **C++** | Zero-Copy / Move | Zero CPU Overhead | Requires Intra-Process |
| **33** | **CUDA** | Kernels | Fast Data Processing | Complex Coding |
| **34** | **RTOS** | Preemption | Low Jitter | System complexity |

---

## 🚀 Weekly Capstone: "The Speed Demon" Pipeline

**Scenario:** Drone Person Follower.
- **Input:** 4K Video.
- **Tasks:** Detect (YOLO), Track (DeepSORT), Control (PID).
- **Constraint:** Must run at 30Hz on embedded hardware. 100ms latency budget.

### 🛠️ Project Structure
```text
week5_capstone/
├── src/
│   ├── capture_node.cpp (Zero Copy)
│   ├── infer_node.cpp (TensorRT)
│   ├── control_node.cpp (Real-Time)
│   └── launch/system.launch.py
└── lib/
    └── libyolo_trt.so
```

### 👨‍💻 Component 1: Zero-Copy Capture (`capture_node.cpp`)

Captures from GStreamer (Hardware Decoder) directly into a `unique_ptr` buffer compatible with CUDA.

```cpp
// ... Capture logic ...
void on_frame(const cv::Mat& frame) {
    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    // Assume frame data is pinned or mapped
    msg->data = frame.data; 
    pub_->publish(std::move(msg));
}
```

### 👨‍💻 Component 2: TensorRT Inference (`infer_node.cpp`)

Receives pointer. Passes CUDA pointer to TensorRT Engine.

```cpp
void on_image(const sensor_msgs::msg::Image::UniquePtr msg) {
    // 1. Get CUDA Pointer to msg data (Zero Copy)
    void* input_ptr = get_cuda_ptr(msg.get());
    
    // 2. Execute Async (TensorRT)
    // Non-blocking call. Returns immediately.
    context_->enqueueV2(&input_ptr, stream_, nullptr);
    
    // 3. Callback when done
    cudaStreamAddCallback(stream_, on_inference_done, this, 0);
}
```

### 👨‍💻 Component 3: Real-Time Control Loop (`control_node.cpp`)

Runs on a dedicated high-priority thread (`SCHED_FIFO`).
Uses the *latest* tracking box available. If Perception lags, it extrapolated the box (Kalman Filter) to maintain 100Hz control output.

```cpp
void control_loop() { // 100Hz Timer
    // Read Shared Memory (Lock Free)
    State target = atomic_load(latest_target);
    
    // Kalman Prediction (Fill in the gaps)
    target = predict_state(target, dt);
    
    // PID
    cmd = compute_pid(target);
    publish_command(cmd);
}
```

---

## 📝 Self-Assessment Quiz

1.  **Bottlenecks:**
    *   If GPU usage is 50% but FPS is low, what is wrong?
        *   **A:** CPU bottleneck (pre-processing?) or Memory Bandwidth (copying images?). Use `tegrastats` or Nsight Systems to visualize.

2.  **Concurrency:**
    *   Why use `cudaStream`?
        *   **A:** To overlap Copy and Compute. While the GPU computes Frame N, the CPU can copy Frame N+1. This pipelines the workload.

3.  **Safety:**
    *   What happens if TensorRT crashes (Segfault)?
        *   **A:** The whole process dies. In a Composition (Intra-process), the Control Node dies too.
        *   **Mitigation:** Watchdog Timer (Hardware) resets the board. Or run Control in a separate process (Safety Island) with Inter-process comms.

---

## ⏭️ Look Ahead: Week 6
We have perception, planning, control, and optimization. Now we need to **Collaborate**.
**Week 6: Multi-Robot Systems.**
*   Swarm Intelligence.
*   Centralized vs Decentralized Architectures.
*   Cloud Robotics (AWS RoboMaker).
*   Shared Maps.

---

**Week 5 Complete**
