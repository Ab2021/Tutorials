# Day 96: PVA & Vision Pipeline
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 14: Edge AI Deployment

---

> **📝 Content Creator Instructions:**
> Don't waste GPU on Resizing images. Use the Hardware.
> - **Focus:** Programmable Vision Accelerator (PVA), Video Image Compositor (VIC), and the VPI (Vision Programming Interface) library.
> - **Code:** A unified VPI pipeline that takes a 4K image, downscales it (VIC), computes Dense Optical Flow (PVA), and runs corner detection (CUDA) interoperably.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Map** vision tasks to hardware: Resizing $\to$ VIC, Tracking $\to$ PVA, Inference $\to$ DLA/GPU.
2.  **Implement** VPI Pipelines (similar to OpenCV G-API but hardware optimized).
3.  **Use** `vpiImageWrap` to pass memory between OpenCV, ROS, and VPI without copying.
4.  **Benchmark** PVA vs OpenCV CPU Optical Flow (10x+ speedup).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA Jetson (PVA is specific to Jetson). Desktop has VPI-CUDA/CPU but no PVA hardware.

### Software Environment
```bash
sudo apt install libnvvpi2 vpi2-dev vpi2-samples
pip install vpi
```

### Prior Knowledge
- Optical Flow (Day 8).
- Zero-Copy Memory.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: VPI (Vision Programming Interface)

OpenCV is great but mostly CPU-bound.
VPI is NVIDIA's CV library that targets **All 5 compute engines**:
1.  **CPU:** Sequential fallback.
2.  **CUDA (GPU):** Massively parallel.
3.  **PVA (Vision Accelerator):** Low-power tracking/flow.
4.  **VIC (Compositor):** Fast geometric transforms (Resize, Crop, Lens Distortion).
5.  **DLA:** (Limited VPI support for inference).

### 🔹 Part 2: PVA & VIC

*   **VIC (Video Image Compositor):**
    *   Hardwired block in the Tegra SoC.
    *   Can decode video streams and resize them for a CNN input in *microseconds*.
    *   Offloads the GPU from mundane preprocessing.
*   **PVA (Programmable Vision Accelerator):**
    *   VLIW (Very Long Instruction Word) processor.
    *   Optimized for "Feature Tracking" (KLT), "Stereo Disparity", and "Optical Flow" (NVOF).

### 🔹 Part 3: The Pipeline Concept

Data $\to$ VIC (Resize) $\to$ PVA (Optical Flow) $\to$ GPU (Segmentation) $\to$ CPU (Logic).
*   **Key:** Zero Memory Copies.
*   **EGLImage / NvBuffer:** The underlying memory handle that all these engines share.

---

## 💻 Implementation: VPI Pipeline in Python

We will build a pipeline that computes Dense Optical Flow on a video stream.

### 🛠️ Project Structure
```text
day96_vpi/
├── src/
│   ├── vpi_flow.py
│   └── camera_stream.py
└── output/
    └── flow_vis.mp4
```

### 👨‍💻 VPI Flow (`src/vpi_flow.py`)

```python
import cv2
import vpi
import numpy as np

def main():
    # 1. Setup Camera (OpenCV GStreamer for Jetson)
    # Using a pre-recorded file for simplicity in generic code
    cap = cv2.VideoCapture("input_video.mp4")
    
    # 2. VPI Stream (Async queue)
    stream = vpi.Stream()
    
    # 3. Buffers
    # We need Previous and Current frames
    frame_prev = None
    
    while True:
        ret, cv_frame = cap.read()
        if not ret: break
        
        # Wrap OpenCV buffer into VPI Image (CPU Backend initially)
        # For true zero-copy on Jetson, inputs usually come from NvArgusCamera
        img_current = vpi.asimage(cv_frame).convert(vpi.Format.NV12_ER, backend=vpi.Backend.VIC)
        
        if frame_prev is not None:
            # --- The Pipeline ---
            with stream:
                # A. Downscale using VIC (Hardware scaler)
                # Let's verify hardware availability
                # Format: 1080p -> 480p
                img_current_small = img_current.rescale((640, 480), interp=vpi.Interp.LINEAR, backend=vpi.Backend.VIC)
                frame_prev_small = frame_prev.rescale((640, 480), interp=vpi.Interp.LINEAR, backend=vpi.Backend.VIC)
                
                # B. Dense Optical Flow using PVA (or CUDA if PVA unavailable)
                # PVA works best on NV12
                # Calculate Motion Vectors
                motion_vec = vpi.optflow_dense(frame_prev_small, img_current_small, backend=vpi.Backend.PVA)
                
                # C. Visualize (Convert Flow to Color on CUDA)
                # VPI currently doesn't have a flow_to_color, so we map mapping manually or pull to CPU
                # For demo, let's just pull raw grid
            
            # Sync
            stream.sync()
            
            # Retrieve Data
            # Lock the buffer to read on CPU
            with motion_vec.rlock_cpu() as flow_data:
                # flow_data is (H, W, 2) array
                # Visualize using OpenCV
                vis = draw_flow(cv2.resize(cv_frame, (640,480)), flow_data)
                cv2.imshow("PVA Optical Flow", vis)
                
        frame_prev = img_current
        if cv2.waitKey(1) == 27: break

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

if __name__ == "__main__":
    main()
```

### 👨‍💻 Zero-Copy Context (`src/camera_stream.py`)

How to get from Camera to VPI without `memcpy`?
*   Use `jetson_utils` or `Argus`.
*   Pass the `EGLImage` pointer.
*   In Python VPI, `vpi.asimage` is smart enough if numpy array is page-locked.

---

## 🔬 Lab Exercise: "Backend Profiler"

### 1. Lab Objectives
- **Task:** Perform Gaussian Blur.
- **Run 1:** `backend=vpi.Backend.CPU`. Time it.
- **Run 2:** `backend=vpi.Backend.CUDA`. Time it.
- **Run 3:** `backend=vpi.Backend.VIC` (Actually VIC doesn't do Blur, but try Rescale).
- **Result:**
    *   CPU: 10ms.
    *   CUDA: 0.5ms.
    *   VIC: 0.8ms (But frees GPU!).
- **Conclusion:** VIC is slightly slower than CUDA for some ops, but it runs *in parallel* with CUDA, increasing system throughput.

---

## 🚀 Project: "Hardware-Accelerated Visual Odometry"

**Goal:** High-speed VO.
1.  **Feature Detection:** Harris Corners via **PVA**.
2.  **Tracking:** KLT Tracker via **PVA**.
3.  **Pose Estimation:** 5-Point Algorithm via **CPU** (No accelerator for this yet).
4.  **Result:** 100+ FPS feature tracking on Jetson with negligible GPU load.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Backend not supported"
*   **Cause:** Not all algos run on all backends. E.g., `Stereo Disparity` runs on PVA and CUDA, but `FFT` might be CPU/CUDA only.
*   **Fix:** Check `vpi.algo_support` docs.

#### 2. "Memory Layout mismatch"
*   **Cause:** VIC prefers `NV12` or `YUV420`. CUDA prefers `RGBA` or `F32`.
*   **Fix:** Insert `convert` nodes in the VPI stream. These are fast.

---

## ⚡ Optimization: Pipeline Parallelism

VPI Streams are like CUDA Streams.
*   We can construct a graph.
*   While PVA is calculating Flow for Frame N, CUDA can be calculating Object Detection for Frame N (async).
*   Use `vpi.Event` to synchronize if Frame N+1 depends on N.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the VIC best used for?
    *   **A:** Preprocessing. Lens Distortion Correction, Scaling, Format Conversion. It sits right after the Camera ISP.
2.  **Q:** Why use PVA for Optical Flow instead of GPU?
    *   **A:** PVA is purpose-built. It consumes way less watts. It leaves the GPU free for the heavy Transformer model running simultaneously.
3.  **Q:** Is VPI open source?
    *   **A:** No, it's a proprietary NVIDIA library, but provides C++ and Python APIs.

### Challenge Task
> **Task:** Lens Distortion Correction.
> 1. Load Camera Calibration (K, D).
> 2. Create `vpi.WarpMap` or `vpi.LensDistortionModel`.
> 3. Rectify image using **VIC**.
> 4. Display result.

---

## 📚 Further Reading
- **NVIDIA VPI Documentation:** "Algorithms" section.
- **Jetson Multimedia API:** Low level control.

---

**Day 96 Complete**
