# Day 107: Doppler Estimation (2D FFT)
## Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing

---

> **📝 Day 107 Focus:**
> Knowing "There is a car at 50m" is good. Knowing "It is approaching at 100 km/h" is better. **Doppler Estimation** uses the phase shift across multiple chirps to measure velocity. This leads to the **Range-Doppler Map**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** how phase shift relates to small movements (Doppler).
2.  **Structure** the Radar Data Cube (Fast Time vs. Slow Time).
3.  **Implement** 2D FFT (Range FFT + Doppler FFT).
4.  **Visualize** a Range-Doppler Map (Heatmap).
5.  **Calculate** Max Velocity and Velocity Resolution.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 106:** FMCW Range FFT.
-   **Math:** $e^{j\omega t}$ (Complex Exponentials).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Phase Shift

When an object moves slightly between two chirps (separated by $T_c$), the range changes by $\Delta R = v \cdot T_c$.
-   This $\Delta R$ is too small to see in the Range FFT.
-   However, it causes a **Phase Shift** $\Delta \phi = \frac{4\pi \Delta R}{\lambda}$.
-   Since $\lambda$ (wavelength) is tiny (4mm at 77GHz), even mm-level movement causes a measurable phase rotation.

### 🔹 Part 2: The Radar Data Cube

We transmit a **Frame** consisting of $N_{chirps}$ (e.g., 128 chirps).
-   **Fast Time ($t$):** Samples within one chirp (Range info).
-   **Slow Time ($m$):** Index of the chirp (Velocity info).
-   **Data Matrix:** Size $[N_{samples} \times N_{chirps}]$.

### 🔹 Part 3: 2D FFT

1.  **Range FFT:** FFT along rows (Fast Time). Result: Range bins.
2.  **Doppler FFT:** FFT along columns (Slow Time). Result: Velocity bins.
3.  **Output:** A 2D matrix where X-axis is Velocity and Y-axis is Range.

---

## 💻 Implementation: Range-Doppler Map

**Scenario:**
-   **Target 1:** Range 50m, Velocity 10 m/s (Approaching).
-   **Target 2:** Range 30m, Velocity -5 m/s (Receding).
-   **Task:** Generate Range-Doppler Map.

### 🛠️ Setup
Create `week16_day107` and `doppler_sim.py`.

```bash
mkdir -p ~/ros2_ws/src/week16_day107
cd ~/ros2_ws/src/week16_day107
touch doppler_sim.py
```

### 👨‍💻 Code: 2D FFT Processing

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Radar Config ---
fc = 77e9
c = 3e8
lambda_ = c / fc

Tc = 40e-6          # Chirp Duration
B = 4e9             # Bandwidth
S = B / Tc          # Slope
fs = 10e6           # Sample Rate
Ns = 256            # Samples per Chirp
Nc = 128            # Number of Chirps (Frame)

def simulate_frame():
    # Time vectors
    t = np.linspace(0, Tc, Ns) # Fast time
    chirp_indices = np.arange(Nc) # Slow time
    
    # Data Cube: [Ns, Nc]
    # We simulate the IF signal for each chirp
    mix_signal = np.zeros((Ns, Nc), dtype=complex)
    
    # --- Targets ---
    # Range (m), Velocity (m/s)
    targets = [
        (50.0, 20.0),  # 50m, 20m/s (Closing)
        (30.0, -10.0)  # 30m, -10m/s (Opening)
    ]
    
    for r0, v in targets:
        for m in range(Nc):
            # Range at chirp m
            # r(t) = r0 + v * t_total
            # t_total = m * Tc
            r_m = r0 + v * (m * Tc)
            
            # Phase Shift due to Range
            # phi = 2 * pi * (S * 2R/c) * t + 2 * pi * (fc * 2R/c)
            # The first term is beat freq (Range). The second is Doppler phase.
            
            tau = 2 * r_m / c
            beat_freq = S * tau
            doppler_phase = 2 * np.pi * fc * tau
            
            # Signal for this chirp
            # A * exp(j * (2*pi*f_beat*t + phase))
            # Note: We use complex signal for simplicity
            sig = np.exp(1j * (2 * np.pi * beat_freq * t + doppler_phase))
            
            mix_signal[:, m] += sig
            
    # Add Noise
    mix_signal += (np.random.randn(Ns, Nc) + 1j * np.random.randn(Ns, Nc)) * 0.1
    
    return mix_signal

def process_2d_fft(mix_signal):
    # 1. Range FFT (Axis 0)
    range_fft = np.fft.fft(mix_signal, axis=0)
    
    # 2. Doppler FFT (Axis 1)
    # Shift zero frequency to center
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)
    
    # Magnitude
    mag = np.abs(doppler_fft)
    mag_db = 20 * np.log10(mag + 1e-9)
    
    return mag_db

def plot_rd_map(mag_db):
    # Axes Calculation
    # Range Axis
    range_res = c / (2 * B)
    max_range = range_res * Ns
    ranges = np.linspace(0, max_range, Ns)
    
    # Velocity Axis
    # V_max = lambda / (4 * Tc)
    v_max = lambda_ / (4 * Tc)
    velocities = np.linspace(-v_max, v_max, Nc)
    
    plt.figure(figsize=(10, 8))
    # Extent: [xmin, xmax, ymin, ymax]
    # Note: imshow origin is top-left, we want bottom-left
    plt.imshow(mag_db, aspect='auto', origin='lower',
               extent=[-v_max, v_max, 0, max_range],
               cmap='jet')
    
    plt.title("Range-Doppler Map")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.colorbar(label="Power (dB)")
    plt.ylim(0, 100) # Zoom range
    plt.grid(alpha=0.3)
    
    # Mark Targets
    plt.scatter([20, -10], [50, 30], c='white', marker='x', label='Ground Truth')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    data = simulate_frame()
    rd_map = process_2d_fft(data)
    plot_rd_map(rd_map)
```

---

## 🔬 Lab Exercise: The Speed Trap

### Lab Objectives
1.  Run the script.
2.  **Observation:** You see two bright hotspots on the heatmap.
    -   One at Range=50, Vel=20.
    -   One at Range=30, Vel=-10.
3.  **Experiment:**
    -   Set Velocity to `v_max + 5`.
    -   **Result:** **Velocity Aliasing**. The target wraps around to the negative side.
    -   $V_{max} = \frac{\lambda}{4 T_c}$.
    -   To increase $V_{max}$, you must decrease $T_c$ (Faster chirps).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Spectral Leakage (Again)
**Symptom:** "Cross" pattern around the target (vertical/horizontal lines).
**Cause:** Rectangular windowing in both Range and Doppler domains.
**Solution:** Apply 2D Windowing.
    -   `window = np.outer(np.hanning(Ns), np.hanning(Nc))`
    -   `mix_signal *= window`

#### 2. Static Clutter
**Symptom:** Huge blob at Velocity = 0.
**Cause:** Reflections from ground, rails, trees.
**Solution:** **Clutter Removal**. Subtract the mean across the Doppler dimension.
    -   `mix_signal -= np.mean(mix_signal, axis=1, keepdims=True)`

---

## ⚡ Optimization & Best Practices

### 1. Hardware Acceleration
2D FFT on 128x256 matrix is heavy for a CPU.
-   **DSP/HWA:** Radar chips (TI AWR1642) have a Hardware Accelerator (HWA) dedicated to FFT.
-   **GPU:** Use CUDA (`cuFFT`) if processing on a Jetson.

### 2. Memory Layout
-   Radar data comes row-by-row (Chirps).
-   Range FFT is easy (Row-wise).
-   Doppler FFT requires **Transpose** (Column-wise access). This is a memory bottleneck.
-   **Corner Turn:** A specific memory operation to transpose the matrix efficiently in L3 cache.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What determines the Velocity Resolution?
    *   **A:** The total Frame Time ($N_c \times T_c$). Longer observation time = finer velocity steps.
2.  **Q:** Why is there a trade-off between Max Range and Max Velocity?
    *   **A:** Not directly.
        -   Max Range depends on $f_s$ and Slope.
        -   Max Velocity depends on $T_c$.
        -   However, $T_c$ limits the Slope for a given Bandwidth ($S = B/T_c$). So yes, fast chirps (high $V_{max}$) mean steeper slope or lower bandwidth.
3.  **Q:** What is the "Micro-Doppler" effect?
    *   **A:** Small vibrations (e.g., a pedestrian swinging arms) cause sidebands in the Doppler spectrum. Used to classify Pedestrian vs Car.

### Challenge Task
**Task:** Max Velocity Calculation.
1.  $\lambda = 4$ mm. $T_c = 40 \mu s$.
2.  $V_{max} = \frac{0.004}{4 \times 40 \times 10^{-6}} = \frac{0.004}{160 \times 10^{-6}} = 25$ m/s (90 km/h).
3.  Is this enough for Autobahn? No.
4.  How to fix? Reduce $T_c$ to $20 \mu s$.

---

## 📚 Further Reading & References
-   [Radar Signal Processing (Richards)](https://www.amazon.com/Fundamentals-Radar-Signal-Processing-Richards/dp/0071844914)
-   [Matlab Phased Array System Toolbox](https://www.mathworks.com/help/phased/index.html)

---

**Day 107 Complete** | Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing
