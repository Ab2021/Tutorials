# Day 106: FMCW Radar Basics
## Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing

---

> **📝 Day 106 Focus:**
> Cameras fail in fog. LiDAR fails in rain. **Radar** works everywhere. But Radar data isn't just "pixels". It's a complex signal of frequencies and phases. Today, we decode the **FMCW (Frequency Modulated Continuous Wave)** signal to measure distance.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the principle of FMCW Radar (Chirp, Bandwidth, Slope).
2.  **Generate** a Chirp signal in Python.
3.  **Mix** Transmitted (Tx) and Received (Rx) signals to get the Beat Frequency.
4.  **Calculate** Range from the Beat Frequency using FFT.
5.  **Analyze** the trade-off between Bandwidth and Range Resolution.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Signal Processing:** FFT (Fast Fourier Transform).
-   **Physics:** Speed of light ($c = 3 \times 10^8$ m/s).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `scipy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Chirp

A "Chirp" is a sine wave whose frequency increases linearly with time.
-   **Start Frequency ($f_c$):** e.g., 77 GHz.
-   **Bandwidth ($B$):** e.g., 4 GHz (sweeps from 77 to 81 GHz).
-   **Chirp Duration ($T_c$):** e.g., 40 $\mu s$.
-   **Slope ($S$):** $S = B / T_c$.

### 🔹 Part 2: Range Estimation

1.  Radar transmits a chirp ($Tx$).
2.  Signal hits a car at distance $R$ and reflects back ($Rx$).
3.  The round-trip time is $\tau = 2R/c$.
4.  By the time $Rx$ arrives, the $Tx$ frequency has increased.
5.  The difference ($f_{beat} = f_{Tx} - f_{Rx}$) is constant!
6.  **Formula:** $R = \frac{c \cdot f_{beat}}{2S}$.

### 🔹 Part 3: Resolution

-   **Range Resolution ($d_{res}$):** The ability to distinguish two close objects.
-   **Formula:** $d_{res} = \frac{c}{2B}$.
-   **Insight:** More Bandwidth = Better Resolution.
    -   $B = 4$ GHz -> $d_{res} = 3.75$ cm. (Automotive Radar).
    -   $B = 200$ MHz -> $d_{res} = 75$ cm. (Air Traffic Control).

---

## 💻 Implementation: FMCW Simulator

**Scenario:**
-   **Target:** Car at 50m.
-   **Radar:** 77 GHz, 4 GHz Bandwidth.
-   **Task:** Simulate Tx/Rx mixing and find the range via FFT.

### 🛠️ Setup
Create `week16_day106` and `fmcw_sim.py`.

```bash
mkdir -p ~/ros2_ws/src/week16_day106
cd ~/ros2_ws/src/week16_day106
touch fmcw_sim.py
```

### 👨‍💻 Code: 1D Range FFT

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# --- Radar Configuration ---
fc = 77e9           # Carrier Freq (77 GHz)
B = 4e9             # Bandwidth (4 GHz)
Tc = 40e-6          # Chirp Duration (40 us)
slope = B / Tc      # Slope (Hz/s)
fs = 20e6           # Sampling Rate (20 MHz) - ADC
Ns = int(fs * Tc)   # Number of Samples per Chirp
c = 3e8             # Speed of Light

def simulate_fmcw():
    t = np.linspace(0, Tc, Ns)
    
    # --- Target Setup ---
    target_range = 50.0 # meters
    tau = 2 * target_range / c # Round trip delay
    
    # --- Signal Generation ---
    # Tx: cos(2*pi * (fc*t + 0.5*slope*t^2))
    # Rx: Tx delayed by tau
    # Mixer Output (IF Signal): cos(2*pi * (slope*tau*t + fc*tau))
    # We simulate the IF signal directly (Beat Frequency)
    
    # f_beat = S * tau = S * (2R/c)
    f_beat = slope * tau
    print(f"Target Range: {target_range} m")
    print(f"Expected Beat Freq: {f_beat/1e6:.2f} MHz")
    
    # Generate IF Signal (Intermediate Frequency)
    # Ideally: signal = A * cos(2*pi*f_beat*t + phase)
    # Add some noise
    noise = np.random.normal(0, 0.5, Ns)
    if_signal = np.cos(2 * np.pi * f_beat * t) + noise
    
    # --- Range FFT ---
    # Perform FFT to find frequency components
    fft_out = fft(if_signal)
    fft_mag = np.abs(fft_out[:Ns//2]) # Positive half
    
    # Frequency Axis
    freqs = np.linspace(0, fs/2, Ns//2)
    
    # Convert Frequency to Range
    # R = (c * f) / (2 * S)
    ranges = (c * freqs) / (2 * slope)
    
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t*1e6, if_signal)
    plt.title("IF Signal (Time Domain)")
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(ranges, fft_mag)
    plt.title("Range Profile (FFT)")
    plt.xlabel("Range (m)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.xlim(0, 100) # Zoom in
    
    # Find Peak
    peak_idx = np.argmax(fft_mag)
    measured_range = ranges[peak_idx]
    print(f"Measured Range: {measured_range:.2f} m")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_fmcw()
```

---

## 🔬 Lab Exercise: The Resolution Test

### Lab Objectives
1.  Run the script.
2.  **Observation:** Peak at exactly 50m.
3.  **Experiment:**
    -   Add a second target at 52m.
    -   `if_signal = cos(f_beat1) + cos(f_beat2)`.
    -   **Result:** Can you see two distinct peaks?
    -   **Calculation:** $d_{res} = c / 2B = 3e8 / 8e9 = 0.0375$ m.
    -   Since $52 - 50 = 2m > 0.0375m$, you should see two peaks clearly.
    -   Try 50.01m. They will merge into one blob.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Aliasing
**Symptom:** Target appears at wrong range (Ghost).
**Cause:** Beat frequency > Nyquist Frequency ($f_s / 2$).
**Solution:** Increase Sampling Rate ($f_s$) or reduce Max Range.
    -   $R_{max} = \frac{f_s \cdot c}{2S}$.

#### 2. Leakage
**Symptom:** Peak is wide and has "side lobes".
**Cause:** Finite sampling window (Rectangular Window).
**Solution:** Apply a **Window Function** (Hanning/Hamming) before FFT.
    -   `fft(if_signal * np.hanning(Ns))`

---

## ⚡ Optimization & Best Practices

### 1. Zero Padding
FFT resolution depends on $N$.
-   If $N=1024$, bins are coarse.
-   **Zero Padding:** Append zeros to the signal to increase $N$ to 4096.
-   This interpolates the spectrum, giving a smoother curve and more accurate peak location.

### 2. CFAR (Constant False Alarm Rate)
In real radar, noise floor varies.
-   A fixed threshold (e.g., `mag > 100`) is bad.
-   **CFAR:** Adaptive threshold based on local noise average. (We will cover this in Day 108).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we use FMCW instead of Pulse Radar?
    *   **A:** FMCW transmits continuously (lower peak power, harder to detect) and measures range via frequency (very accurate). Pulse radar needs high power short pulses.
2.  **Q:** What happens if I double the Bandwidth?
    *   **A:** Range Resolution improves by 2x (Peak becomes sharper).
3.  **Q:** What is the "Beat Frequency"?
    *   **A:** The frequency difference between the Transmitted chirp and the Received chirp. It is proportional to Range.

### Challenge Task
**Task:** Max Range Calculation.
1.  Given $f_s = 10$ MHz, $S = 100$ MHz/$\mu s$.
2.  Calculate $R_{max}$.
3.  $f_{beat\_max} = f_s / 2 = 5$ MHz.
4.  $R_{max} = \frac{c \cdot f_{beat\_max}}{2S}$.
5.  Plug in numbers.

---

## 📚 Further Reading & References
-   [TI mmWave Radar Training](https://training.ti.com/mmwave-training-series)
-   [Radartutorial.eu](https://www.radartutorial.eu/02.basics/Frequency%20Modulated%20Continuous%20Wave%20Radar.en.html)

---

**Day 106 Complete** | Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing
