# Day 92: NVIDIA Jetson Platform Deep Dive
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 14: Edge AI Deployment

---

> **📝 Content Creator Instructions:**
> Cloud is too slow. Deployment happens on the Edge.
> - **Focus:** The Jetson Architecture (Orin AGX/Nano), JetPack SDK, Power Modes (NVPModel), and Hardware Components (DLA, PVA).
> - **Code:** A system monitor script that parses `tegrastats` to log GPU/CPU usage and Throttling events.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Jetson Orin Nano, NX, and AGX (TOPS vs Power).
2.  **Explain** the Heterogeneous Compute blocks: CPU (ARM), GPU (Ampere), DLA (Deep Learning Accelerator), PVA (Vision), VIC (Image Compositor).
3.  **Configure** Power Modes (`nvpmodel`) to balance Performance vs Battery Life.
4.  **Monitor** thermal throttling and voltage rails.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA Jetson (Orin preferred, Xavier/Nano works) OR detailed study of architecture if hardware n/a.

### Software Environment
```bash
# On Jetson
sudo apt install tegra-stats-tools
pip install jtop # Jetson Stats
```

### Prior Knowledge
- Linux Command Line.
- Python Basics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Heterogeneous SoC

A Desktop GPU is just a GPU. A Jetson is a System-on-Chip (SoC).
*   **CPU (ARM Cortex-A78AE):** General logic, ROS 2 nodes, Serial drivers.
*   **GPU (Ampere):** CUDA, Training, Heavy Inference.
*   **DLA (Deep Learning Accelerator):** Specialized ASIC for CNNs (ResNet/YOLO). simpler, lower power, but fixed function.
*   **PVA (Programmable Vision Accelerator):** 7-DOF tracking, Optical Flow (VPI).
*   **VIC (Video Image Compositor):** Lens distortion correction, resizing, color conversion (YUV->RGB).

### 🔹 Part 2: JetPack SDK

The OS + Libraries.
*   **L4T (Linux for Tegra):** Ubuntu 22.04 with custom Kernel.
*   **CUDA, cuDNN, TensorRT:** Accelerated libraries.
*   **VPI (Vision Programming Interface):** Computer Vision library that automatically targets PVA/VIC/GPU.

### 🔹 Part 3: Power Management

Robots run on batteries.
*   **NVPModel:** Defines active cores and clock limits.
    *   `MAXN`: All cores, Max clocks (60W).
    *   `30W`: 4 cores, Reduced clocks.
    *   `15W`: 2 cores, efficient.
*   **Jetson Clocks:** `sudo jetson_clocks` forces max frequency (fan noise up, latency down).

---

## 💻 Implementation: Custom Tegrastats Logger

We will write a Python class to parse the cryptic `tegrastats` output.
Output Example: `RAM 1903/7765MB (lfb 1134x4MB) SWAP 0/3882MB (cached 0MB) CPU [3%@1190,0%@1190,0%@1190,0%@1190] EMC_FREQ 0% GR3D_FREQ 0% PLL@35.5C CPU@38.5C PMIC@50C GPU@37C AO@46C thermal@38.2C POM_5V_IN 1840/1840 POM_5V_GPU 80/80 ...`

### 🛠️ Project Structure
```text
day92_jetson/
├── src/
│   ├── jetson_monitor.py
│   └── power_manager.py
└── logs/
    └── thermal_log.csv
```

### 👨‍💻 Monitor (`src/jetson_monitor.py`)

```python
import subprocess
import re
import csv
import time
import signal
import sys

class JetsonMonitor:
    def __init__(self, log_file="logs/thermal_log.csv"):
        self.process = subprocess.Popen(
            ['tegrastats', '--interval', '1000'], 
            stdout=subprocess.PIPE, 
            universal_newlines=True
        )
        self.log_file = log_file
        self.running = True
        
        # Regex (Simplified)
        self.re_ram = re.compile(r"RAM (\d+)/(\d+)MB")
        self.re_gpu = re.compile(r"GR3D_FREQ (\d+)%")
        self.re_temp = re.compile(r"GPU@([\d\.]+)C")
        self.re_power = re.compile(r"POM_5V_IN (\d+)/(\d+)") # Instant/Avg mW

    def run(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "RAM_Used", "GPU_Load", "GPU_Temp", "Power_mW"])
            
            try:
                for line in self.process.stdout:
                    if not self.running: break
                    
                    data = self.parse_line(line)
                    if data:
                        writer.writerow(data)
                        print(f"Stats: {data}")
                        
            except KeyboardInterrupt:
                self.cleanup()

    def parse_line(self, line):
        try:
            ram = self.re_ram.search(line).group(1)
            gpu = self.re_gpu.search(line).group(1)
            temp = self.re_temp.search(line).group(1)
            power = self.re_power.search(line).group(1)
            return [time.time(), ram, gpu, temp, power]
        except AttributeError:
            return None # Some fields might be missing in older Jetpacks

    def cleanup(self):
        self.running = False
        self.process.terminate()
        print("Monitor Stopped.")

if __name__ == "__main__":
    mon = JetsonMonitor()
    mon.run()
```

### 👨‍💻 Power Manager (`src/power_manager.py`)

Switch modes via Python.

```python
import subprocess

def set_mode(mode="MAXN"):
    # Map friendly names to IDs (Check /etc/nvpmodel.conf)
    # Typical Orin NX: 0=MAXN, 1=25W, 2=15W
    modes = {"MAXN": 0, "25W": 1, "15W": 2}
    
    if mode not in modes:
        print("Invalid Mode")
        return
        
    cmd = ["sudo", "/usr/sbin/nvpmodel", "-m", str(modes[mode])]
    print(f"Switching to {mode}...")
    subprocess.run(cmd)

def loop_check():
    # Dynamic Frequency Scaling
    # If Temp > 80C, switch to lower power mode
    pass
```

---

## 🔬 Lab Exercise: "The Stress Test"

### 1. Lab Objectives
- **Install:** `jtop` (`pip install jetson-stats`).
- **Run:** `jtop`. View the nice GUI.
- **Task:** 
    1.  Max out CPU (`stress -c 8`). Watch Power draw.
    2.  Max out GPU (Run a CUDA sample or training). Watch Power draw.
    3.  Run both.
- **Observation:**
    *   Does the fan kick in?
    *   Does the clock frequency drop (Thermal Throttling)?
    *   If using a battery, does the voltage sag?

---

## 🚀 Project: "Battery Saver Node"

**Goal:** ROS 2 Node that manages system power.
1.  **Idle:** If `cmd_vel` is zero for 1 minute Use `15W` mode.
2.  **Active:** If `cmd_vel` > 0, Switch to `MAXN`.
3.  **Critical:** If Battery < 10%, Trigger Safety Stop and Shutdown non-essential nodes (simulated).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Permission Denied"
*   **Cause:** Accessing hardware monitoring usually requires `sudo` or `video`/`dialout` group membership.
*   **Fix:** Add user to groups: `sudo usermod -aG video $USER`.

#### 2. "System freezes"
*   **Cause:** Out of RAM. Jetson shares RAM between CPU and GPU (Unified Memory). If you load a 4GB model on a 4GB Nano, the OS crashes.
*   **Fix:** Create a SWAP file (`sudo fallocate ...`). It's slow, but prevents crashes.

---

## ⚡ Optimization: Shared Memory (Zero-Copy)

On Desktop: CPU RAM $\to$ PCIe $\to$ GPU VRAM. Slow.
On Jetson: Unified Memory.
*   **Zero-Copy:** We can allocate memory that both CPU and GPU see. No PCIe transfer.
*   **Code:** `cudaMallocManaged`.
*   Crucial for high-bandwidth Camera processing.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the DLA for?
    *   **A:** Offloading standard CNNs (ResNet) to save GPU for specialized tasks (Transformer/Training). DLA is extremely power efficient.
2.  **Q:** Why not just run everything in `MAXN` mode?
    *   **A:** Battery drain. Heat. Some carrier boards cannot supply 60W current spikes.
3.  **Q:** Difference between Orin and Xavier?
    *   **A:** Orin has Ampere GPU (Tensor Cores for INT8), Xavier has Volta. Orin is ~10x faster for AI.

### Challenge Task
> **Task:** Fan Control.
> 1. Read temperature.
> 2. Write a PID controller for Fan PWM (`/sys/devices/pwm-fan/target_pwm`).
> 3. Standard profile is Step-based. PID provides smoother noise profile.

---

## 📚 Further Reading
- **NVIDIA Docs:** "Jetson Linux Developer Guide".
- **JetsonHacks:** Great blog for practical tips.

---

**Day 92 Complete**
