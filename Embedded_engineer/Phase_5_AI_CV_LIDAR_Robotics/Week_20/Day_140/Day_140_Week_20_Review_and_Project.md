# Day 140: Week 20 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 20: Sim-to-Real & Hardware Acceleration

---

> **📝 Content Creator Instructions:**
> Simulation is the training ground. Hardware is the battlefield.
> - **Goal:** Build the "Ultimate Simulation" pipeline.
> - **Code:** A Unified script `digital_twin.py` that spawns a Parallel Physics environment, runs a CUDA-accelerated perception filter, and communicates via HIL Bridge to a controller.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Orchestrate** multiple acceleration technologies (Parallel Physics + CUDA Perception).
2.  **Validate** systems using Hardware-in-Loop concepts.
3.  **Ensure** Robustness using Domain Randomization.

---

## 📚 Week 20 Review: Faster, Stronger, Virtual

| Day | Topic | Key Lesson | Tool |
|-----|-------|------------|------|
| **134** | **Physics Engines** | Vectorized physics (Batch Sim) beats For-Loops | `Isaac Gym` |
| **135** | **Domain Randomization** | Randomize $\mu, m, K_p$ to cross Reality Gap | `Sim2Real` |
| **136** | **HIL Testing** | Embeddings need real-time feedback loops | `UDP Bridge` |
| **137** | **FPGA** | Hardware logic (Verilog) has near-zero latency | `Verilog` |
| **138** | **CUDA** | GPU is for throughput (Lidar/Vision), not latency | `Numba` |
| **139** | **Cloud** | Offload heavy maps/planning to K8s | `Fog` |

### The Accelerated Pipeline
```mermaid
graph TD
    Sim[Isaac Gym (GPU Physics)] -->|Lidar Data| Filter[CUDA Kernel]
    Filter -->|Filtered Scan| HIL[UDP Bridge]
    HIL -->|State| Controller[Embedded Jetson]
    Controller -->|Motor Cmd| HIL
    HIL -->|Torque| Sim
```

---

## 🚀 Weekly Capstone: "The Digital Twin"

**Scenario:** High-speed Pick and Place.
**Challenge:** Simulate 100 Robots picking objects.
**Components:**
1.  **Sim:** Vectorized Physics (100 instances).
2.  **Perception:** CUDA-based object filter (Remove conveyor belt plane).
3.  **HIL:** Send coordinates to "Robot Controller".

### 🛠️ Project Structure
```text
week20_capstone/
├── src/
│   ├── digital_twin.py
│   ├── accelerator.py (CUDA)
```

### 👨‍💻 Unified Simulation (`src/digital_twin.py`)

A mock-up of the full stack.

```python
import numpy as np
import time
from numba import cuda
import matplotlib.pyplot as plt

# --- 1. CUDA Accelerator ---
@cuda.jit
def fast_filter(points, output_mask, z_limit):
    idx = cuda.grid(1)
    if idx < points.shape[0]:
        if points[idx, 2] > z_limit: # Keep objects above belt
            output_mask[idx] = 1
        else:
            output_mask[idx] = 0

class PerceptionEngine:
    def __init__(self):
        self.threads = 256
        
    def filter_batch(self, point_batch):
        # Flatten batch for simple kernel
        # points: [N_robots, N_points, 3] -> [Total_Pts, 3]
        flat = point_batch.reshape(-1, 3)
        n = flat.shape[0]
        
        blocks = (n + self.threads - 1) // self.threads
        
        d_pts = cuda.to_device(flat)
        d_mask = cuda.device_array(n, dtype=np.uint8)
        
        fast_filter[blocks, self.threads](d_pts, d_mask, 0.1)
        cuda.synchronize()
        
        return d_mask.copy_to_host().reshape(point_batch.shape[0], point_batch.shape[1])

# --- 2. Parallel Simulation ---
class ParallelFactory:
    def __init__(self, n_robots=100):
        self.n = n_robots
        # State: [x, y, z] for End Effector
        self.ee_pos = np.zeros((self.n, 3))
        # Random conveyor objects: [x, y, z] relative to EE
        # Let's say we simulate Lidar points directly
        self.lidar_size = 1000
        
    def step(self, cmds):
        # Vectorized Physics Update
        # cmd: [vx, vy, vz]
        self.ee_pos += cmds * 0.01
        
        # Simulate Sensor Data (Noisy Lidar)
        # Shape: [N_robots, 1000, 3]
        # Most points are Ground (Z=0), some are Object (Z=0.5)
        scans = np.random.normal(0, 0.01, (self.n, self.lidar_size, 3))
        # Add 'Ground'
        scans[:, :, 2] = 0.0 
        # Add 'Object' to first 10 points
        scans[:, :10, 2] = 0.5 
        
        return scans

# --- 3. Main Loop ---
def main():
    N_ROBOTS = 100
    if not cuda.is_available():
        print("GPU needed for Digital Twin Capstone.")
        return

    sim = ParallelFactory(N_ROBOTS)
    perception = PerceptionEngine()
    
    print(f"Starting Digital Twin for {N_ROBOTS} Robots...")
    
    times = []
    
    for t in range(50):
        start = time.time()
        
        # 1. Controller (Host CPU - Mock)
        # Move down
        cmds = np.zeros((N_ROBOTS, 3))
        cmds[:, 2] = -0.1 
        
        # 2. Physics Step
        raw_scans = sim.step(cmds)
        
        # 3. Perception (GPU)
        # Copying 100 * 1000 points to GPU, Filtering, Copying Back
        masks = perception.filter_batch(raw_scans)
        
        # 4. Logic (Count objects)
        # Count how many points surviving filter per robot
        object_points = np.sum(masks, axis=1)
        
        # If > 5 points, Trigger Grasp
        triggers = object_points > 5
        
        end = time.time()
        times.append((end - start)*1000)
        
        if t % 10 == 0:
            print(f"Step {t}: Loop Time={times[-1]:.2f}ms | Robots Triggered: {np.sum(triggers)}")

    print(f"Avg Step Time: {np.mean(times):.2f} ms")
    print(f"Total Throughput: {N_ROBOTS / (np.mean(times)/1000):.0f} Robot-Steps/Sec")
    
    plt.plot(times)
    plt.title("Digital Twin Cycle Time (100 Robots)")
    plt.xlabel("Step")
    plt.ylabel("Time (ms)")
    plt.savefig("output/capstone_perf.png")

if __name__ == "__main__":
    main()
```

---

## 📝 Self-Assessment Quiz

1.  **Architecture:**
    *   Where should the PID control loop run?
    *   **A:** Highest frequency place. FPGA or Real-time Kernel. Not Cloud. Not Python main thread.
2.  **Simulation:**
    *   Why use Vectorized Simulation?
    *   **A:** To train RL agents. RL needs millions of samples. Real-time is too slow. Vectorized allows collecting 1 year of experience in 1 hour.
3.  **Hardware:**
    *   Cost of `cudaMemcpy`?
    *   **A:** High. Avoid moving data back to CPU if possible. Keep it on GPU for the next step (e.g., Feed Lidar directly to PyTorch on GPU).

---

## ⏭️ Look Ahead: Week 21
Robots working with People.
**Week 21: Collaborative Robotics (Cobots).**
*   Safety Standards (ISO 10218).
*   Force/Torque Control.
*   Teaching by Demonstration.

---

**Week 20 Complete**
